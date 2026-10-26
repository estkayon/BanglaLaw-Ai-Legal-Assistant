import json
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger


class DataDriftMonitor:
    """Tracks changes in the legal corpus over time.

    Monitors: number of acts, total chunks, average chunk size,
    and language distribution to detect unexpected shifts.
    """

    SNAPSHOT_FILE = "drift_snapshots.jsonl"

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or Path("logs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_path = self.output_dir / self.SNAPSHOT_FILE

    def compute_snapshot(self, data_dir: Path) -> dict:
        """Compute current data statistics from raw act files."""
        json_files = sorted(data_dir.glob("act_*.json"))

        total_acts = len(json_files)
        total_sections = 0
        total_text_length = 0
        bangla_count = 0
        english_count = 0

        for filepath in json_files:
            try:
                data = json.loads(filepath.read_text(encoding="utf-8"))
                sections = data.get("sections", [])
                total_sections += len(sections)

                for section in sections:
                    text = section.get("section_text", "")
                    total_text_length += len(text)

                    # Simple language detection based on character range
                    bangla_chars = sum(1 for c in text if "\u0980" <= c <= "\u09ff")
                    if bangla_chars > len(text) * 0.3:
                        bangla_count += 1
                    else:
                        english_count += 1
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Skipping {filepath.name}: {e}")
                continue

        avg_section_length = total_text_length / total_sections if total_sections else 0

        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "total_acts": total_acts,
            "total_sections": total_sections,
            "avg_section_length": round(avg_section_length, 1),
            "total_text_chars": total_text_length,
            "bangla_sections": bangla_count,
            "english_sections": english_count,
        }

    def save_snapshot(self, snapshot: dict) -> None:
        """Append snapshot to the JSONL log."""
        with self.snapshot_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
        logger.info(
            f"Drift snapshot saved: {snapshot['total_acts']} acts, {snapshot['total_sections']} sections"
        )

    def load_snapshots(self) -> list[dict]:
        """Load all historical snapshots."""
        if not self.snapshot_path.exists():
            return []
        snapshots = []
        for line in self.snapshot_path.read_text(encoding="utf-8").strip().split("\n"):
            if line:
                snapshots.append(json.loads(line))
        return snapshots

    def detect_drift(self, current: dict, threshold_pct: float = 20.0) -> list[str]:
        """Compare current snapshot against the last one. Returns list of drift warnings."""
        history = self.load_snapshots()
        if not history:
            return []

        previous = history[-1]
        warnings = []

        checks = [
            ("total_acts", "Act count"),
            ("total_sections", "Section count"),
            ("avg_section_length", "Avg section length"),
        ]

        for key, label in checks:
            prev_val = previous.get(key, 0)
            curr_val = current.get(key, 0)

            if prev_val == 0:
                continue

            change_pct = abs(curr_val - prev_val) / prev_val * 100
            if change_pct > threshold_pct:
                direction = "increased" if curr_val > prev_val else "decreased"
                msg = f"{label} {direction} by {change_pct:.1f}% ({prev_val} -> {curr_val})"
                warnings.append(msg)
                logger.warning(f"Data drift: {msg}")

        if not warnings:
            logger.info("No significant data drift detected")

        return warnings

    def run(self, data_dir: Path) -> tuple[dict, list[str]]:
        """Compute snapshot, detect drift, save, and return results."""
        snapshot = self.compute_snapshot(data_dir)
        warnings = self.detect_drift(snapshot)
        self.save_snapshot(snapshot)
        return snapshot, warnings
