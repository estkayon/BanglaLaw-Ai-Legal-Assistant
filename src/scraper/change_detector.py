import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from src.utils.config import settings


@dataclass
class ChangeResult:
    """Result of comparing scraped data against stored hashes."""

    new_files: list[Path]
    modified_files: list[Path]
    unchanged_files: list[Path]

    @property
    def has_changes(self) -> bool:
        return bool(self.new_files or self.modified_files)

    @property
    def summary(self) -> str:
        return (
            f"New: {len(self.new_files)}, "
            f"Modified: {len(self.modified_files)}, "
            f"Unchanged: {len(self.unchanged_files)}"
        )


class ChangeDetector:
    """Detects changes in scraped legal data by comparing file hashes."""

    HASH_FILE = "file_hashes.json"

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir or settings.data_raw_dir
        self.hash_file = self.data_dir / self.HASH_FILE
        self.hashes = self._load_hashes()

    def _load_hashes(self) -> dict[str, str]:
        """Load stored file hashes from disk."""
        if not self.hash_file.exists():
            return {}
        try:
            return json.loads(self.hash_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not load hash file, starting fresh: {e}")
            return {}

    def _save_hashes(self) -> None:
        """Persist current hashes to disk."""
        self.hash_file.parent.mkdir(parents=True, exist_ok=True)
        self.hash_file.write_text(
            json.dumps(self.hashes, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @staticmethod
    def _compute_hash(filepath: Path) -> str:
        """SHA256 hash of file contents."""
        return hashlib.sha256(filepath.read_bytes()).hexdigest()

    def detect_changes(self) -> ChangeResult:
        """Compare current data files against stored hashes."""
        new_files: list[Path] = []
        modified_files: list[Path] = []
        unchanged_files: list[Path] = []

        json_files = sorted(self.data_dir.glob("act_*.json"))

        for filepath in json_files:
            filename = filepath.name
            current_hash = self._compute_hash(filepath)

            if filename not in self.hashes:
                new_files.append(filepath)
            elif self.hashes[filename] != current_hash:
                modified_files.append(filepath)
            else:
                unchanged_files.append(filepath)

        result = ChangeResult(
            new_files=new_files,
            modified_files=modified_files,
            unchanged_files=unchanged_files,
        )

        logger.info(f"Change detection: {result.summary}")
        return result

    def update_hashes(self, files: list[Path] | None = None) -> None:
        """Update stored hashes for given files, or all act files if none specified."""
        targets = files or sorted(self.data_dir.glob("act_*.json"))

        for filepath in targets:
            self.hashes[filepath.name] = self._compute_hash(filepath)

        self._save_hashes()
        logger.info(f"Updated hashes for {len(targets)} files")

    def get_changed_files(self) -> list[Path]:
        """Convenience: detect changes, update hashes, return new + modified files."""
        result = self.detect_changes()
        changed = result.new_files + result.modified_files

        if changed:
            self.update_hashes(changed)

        return changed
