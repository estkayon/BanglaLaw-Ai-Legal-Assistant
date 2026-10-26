import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.config import settings


@dataclass
class LawSection:
    """A single section (ধারা) of a law."""

    section_id: str
    section_number: str
    section_title: str
    section_text: str
    section_url: str


@dataclass
class LawAct:
    """A single act/law with its metadata and sections."""

    act_id: str
    act_title: str
    act_number: str
    act_year: str
    act_date: str
    act_url: str
    volume: str
    preamble: str = ""
    sections: list[LawSection] = field(default_factory=list)


class BDLawsScraper:
    """Scraper for bdlaws.minlaw.gov.bd legal corpus."""

    BASE_URL = settings.scraper_base_url
    DELAY = settings.scraper_delay_seconds

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or settings.data_raw_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client = httpx.Client(
            timeout=30.0,
            follow_redirects=True,
            headers={
                "User-Agent": "BanglaLaw-Research-Bot/1.0 (legal-research; +https://github.com/estkayon/banglalaw-ai-legal-assistant)",
                "Accept-Language": "bn,en;q=0.9",
            },
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def _fetch_page(self, url: str) -> BeautifulSoup:
        """Fetch a page and return parsed HTML."""
        logger.debug(f"Fetching: {url}")
        response = self.client.get(url)
        response.raise_for_status()
        time.sleep(self.DELAY)
        return BeautifulSoup(response.text, "lxml")

    def get_volume_urls(self) -> list[dict[str, str]]:
        """Get all volume page URLs from the main index."""
        soup = self._fetch_page(f"{self.BASE_URL}/laws-of-bangladesh.html")
        volumes = []

        for link in soup.select("a[href*='volume-']"):
            href = link.get("href", "")
            if "volume-" in href and href.endswith(".html"):
                volume_num = href.split("volume-")[1].replace(".html", "")
                volumes.append(
                    {
                        "volume": volume_num,
                        "url": href
                        if href.startswith("http")
                        else f"{self.BASE_URL}/{href.lstrip('/')}",
                        "title": link.get_text(strip=True),
                    }
                )

        # Deduplicate by volume number (sidebar repeats volume links)
        seen = set()
        unique = []
        for v in volumes:
            if v["volume"] not in seen:
                seen.add(v["volume"])
                unique.append(v)

        logger.info(f"Found {len(unique)} volumes")
        return unique

    def get_acts_from_volume(self, volume_url: str, volume_num: str) -> list[dict[str, str]]:
        """Extract act links and metadata from a volume page."""
        soup = self._fetch_page(volume_url)
        acts = []

        # Acts are in table rows with links to act-{id}.html
        for link in soup.select("a[href*='act-']"):
            href = link.get("href", "")
            if "/act-" not in href or "section" in href:
                continue

            act_id = href.split("act-")[1].replace(".html", "")
            if not act_id.isdigit():
                continue

            title = link.get_text(strip=True)
            if not title or len(title) < 2:
                continue

            full_url = href if href.startswith("http") else f"{self.BASE_URL}/{href.lstrip('/')}"

            acts.append(
                {
                    "act_id": act_id,
                    "title": title,
                    "url": full_url,
                    "volume": volume_num,
                }
            )

        # Deduplicate by act_id (title and number links both go to same act)
        seen = set()
        unique = []
        for a in acts:
            if a["act_id"] not in seen:
                seen.add(a["act_id"])
                unique.append(a)

        logger.info(f"Volume {volume_num}: found {len(unique)} acts")
        return unique

    def get_act_details(self, act_url: str) -> dict:
        """Extract act metadata and section links from an act page."""
        soup = self._fetch_page(act_url)

        # Act title from h3
        title_tag = soup.select_one("h3")
        title = title_tag.get_text(strip=True) if title_tag else ""

        # Act number and date from h4 and brackets
        subtitle = soup.select_one("h4")
        act_number = subtitle.get_text(strip=True) if subtitle else ""

        # Preamble text (bold text before section list)
        preamble_parts = []
        for bold in soup.select("strong"):
            text = bold.get_text(strip=True)
            if text and len(text) > 20:
                preamble_parts.append(text)
        preamble = " ".join(preamble_parts)

        # Section links
        sections = []
        for link in soup.select("a[href*='section-']"):
            href = link.get("href", "")
            if "section-" not in href:
                continue

            section_id = href.split("section-")[1].replace(".html", "")
            section_title = link.get_text(strip=True)
            full_url = href if href.startswith("http") else f"{self.BASE_URL}/{href.lstrip('/')}"

            sections.append(
                {
                    "section_id": section_id,
                    "title": section_title,
                    "url": full_url,
                }
            )

        return {
            "title": title,
            "act_number": act_number,
            "preamble": preamble,
            "sections": sections,
        }

    def scrape_section(self, section_url: str) -> dict[str, str]:
        """Scrape the full text of a single section page."""
        soup = self._fetch_page(section_url)

        # Remove navigation, footer, sidebar, footnotes
        for tag in soup.select("nav, footer, .sidebar, .related-links"):
            tag.decompose()

        # Section title from page title or first heading after act title
        page_title = soup.title.get_text(strip=True) if soup.title else ""

        # Main content: everything between the section heading and the footnotes line
        content_parts = []
        main_content = soup.select_one(".container") or soup.select_one("body")

        if main_content:
            # Find the horizontal rule that separates content from footnotes
            for element in main_content.find_all(["p", "div", "span", "table"]):
                # Stop at footnote separator
                if element.find("img", src=lambda s: s and "line" in s):
                    break
                text = element.get_text(strip=True)
                if text and text not in page_title:
                    content_parts.append(text)

        # Fallback: get all paragraph text
        if not content_parts:
            for p in soup.select("p"):
                text = p.get_text(strip=True)
                if text:
                    content_parts.append(text)

        section_text = "\n".join(content_parts)

        # Extract section number from title (e.g., "২৷ সংজ্ঞা" -> "২")
        section_number = ""
        if "|" in page_title:
            section_part = page_title.split("|")[-1].strip()
            # Section number is before the dot or tab
            for char in section_part:
                if char in ("৷", ".", "\t", " "):
                    break
                section_number += char

        return {
            "section_number": section_number,
            "section_title": page_title.split("|")[-1].strip() if "|" in page_title else page_title,
            "section_text": section_text,
        }

    def scrape_act(self, act_info: dict) -> LawAct:
        """Scrape a complete act with all its sections."""
        logger.info(f"Scraping act: {act_info['title']} (ID: {act_info['act_id']})")

        details = self.get_act_details(act_info["url"])

        act = LawAct(
            act_id=act_info["act_id"],
            act_title=details["title"] or act_info["title"],
            act_number=details["act_number"],
            act_year="",
            act_date="",
            act_url=act_info["url"],
            volume=act_info.get("volume", ""),
            preamble=details["preamble"],
        )

        for section_info in details["sections"]:
            try:
                section_data = self.scrape_section(section_info["url"])
                section = LawSection(
                    section_id=section_info["section_id"],
                    section_number=section_data["section_number"],
                    section_title=section_data["section_title"],
                    section_text=section_data["section_text"],
                    section_url=section_info["url"],
                )
                act.sections.append(section)
            except Exception as e:
                logger.warning(f"Failed to scrape section {section_info['section_id']}: {e}")
                continue

        logger.info(f"Scraped {len(act.sections)} sections from: {act.act_title}")
        return act

    def save_act(self, act: LawAct) -> Path:
        """Save a scraped act as JSON."""
        filepath = self.output_dir / f"act_{act.act_id}.json"
        data = asdict(act)
        filepath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.debug(f"Saved: {filepath}")
        return filepath

    def compute_hash(self, filepath: Path) -> str:
        """Compute SHA256 hash of a file for change detection."""
        return hashlib.sha256(filepath.read_bytes()).hexdigest()

    def scrape_volume(self, volume_num: str) -> list[LawAct]:
        """Scrape all acts from a single volume."""
        volume_url = f"{self.BASE_URL}/volume-{volume_num}.html"
        acts_info = self.get_acts_from_volume(volume_url, volume_num)
        scraped_acts = []

        for act_info in acts_info:
            try:
                act = self.scrape_act(act_info)
                self.save_act(act)
                scraped_acts.append(act)
            except Exception as e:
                logger.error(f"Failed to scrape act {act_info['act_id']}: {e}")
                continue

        return scraped_acts

    def scrape_all(self, volume_range: tuple[int, int] | None = None) -> list[Path]:
        """Scrape all volumes or a specific range. Returns list of saved file paths."""
        volumes = self.get_volume_urls()
        saved_files = []

        if volume_range:
            start, end = volume_range
            volumes = [v for v in volumes if start <= int(v["volume"]) <= end]
            logger.info(f"Scraping volumes {start} to {end}")

        for volume in volumes:
            try:
                acts = self.scrape_volume(volume["volume"])
                for act in acts:
                    filepath = self.output_dir / f"act_{act.act_id}.json"
                    if filepath.exists():
                        saved_files.append(filepath)
            except Exception as e:
                logger.error(f"Failed to scrape volume {volume['volume']}: {e}")
                continue

        logger.info(f"Scraping complete. {len(saved_files)} acts saved.")
        return saved_files
