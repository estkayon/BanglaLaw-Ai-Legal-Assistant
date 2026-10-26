"""Entry point for the scheduled scraper workflow."""

from loguru import logger

from src.scraper.bdlaws_scraper import BDLawsScraper
from src.scraper.change_detector import ChangeDetector


def main() -> None:
    logger.info("Starting scheduled scrape...")

    with BDLawsScraper() as scraper:
        scraper.scrape_all()

    detector = ChangeDetector()
    changed = detector.get_changed_files()

    if changed:
        logger.info(f"Found {len(changed)} new/modified files")
    else:
        logger.info("No changes detected")


if __name__ == "__main__":
    main()
