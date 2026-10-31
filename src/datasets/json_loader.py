"""
JSON / JSONL dataset loader.

Expects a JSON file with one of two formats:

1. Array of objects:
   [
     {"id": "1", "text": "...", "book": "..."},
     ...
   ]

2. JSONL (one JSON object per line):
   {"id": "1", "text": "..."}
   {"id": "2", "text": "..."}

Required fields per record: id, text.
All other fields are optional and default to None if absent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Dict

from src.datasets.base import Hadith, HadithDataset
from src.logging.logger import get_logger

logger = get_logger(__name__)


class JSONHadithDataset(HadithDataset):
    """
    Load Hadiths from a JSON or JSONL file.

    Args:
        path: Path to the JSON/JSONL file.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._hadiths: List[Hadith] = []
        self._id_map: Dict[str, Hadith] = {}
        self._loaded = False

    def load(self) -> None:
        """
        Parse the dataset file.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If records are missing required fields.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.path}")

        logger.info("Loading dataset from %s", self.path)

        raw_records = self._parse_file()
        self._hadiths = []
        parse_errors = 0

        for i, record in enumerate(raw_records):
            try:
                hadith = self._parse_record(record, index=i)
                self._hadiths.append(hadith)
                self._id_map[hadith.id] = hadith
            except (KeyError, ValueError) as exc:
                parse_errors += 1
                logger.error(
                    "Failed to parse record %d: %s — skipping. Error: %s",
                    i, str(record)[:120], exc
                )

        self._loaded = True
        logger.info(
            "Loaded %d Hadiths (%d parse errors) from %s",
            len(self._hadiths), parse_errors, self.path
        )

    def _parse_file(self) -> List[dict]:
        """Read JSON array or JSONL file and return list of raw dicts."""
        content = self.path.read_text(encoding="utf-8").strip()

        # Try JSON array first
        if content.startswith("["):
            return json.loads(content)

        # Try JSONL
        records = []
        for line_num, line in enumerate(content.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.error("JSON parse error on line %d: %s", line_num, exc)
        return records

    @staticmethod
    def _parse_record(record: dict, index: int) -> Hadith:
        """Convert a raw dict to a Hadith dataclass."""
        if "text" not in record:
            raise ValueError(f"Record {index} missing required field 'text'")

        hadith_id = str(record.get("id", str(index)))
        return Hadith(
            id=hadith_id,
            text=record["text"],
            book=record.get("book"),
            chapter=record.get("chapter"),
            narrator=record.get("narrator"),
            source=record.get("source"),
            reference=record.get("reference"),
        )

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("Dataset not loaded. Call load() first.")

    def get_texts(self) -> List[str]:
        self._require_loaded()
        return [h.text for h in self._hadiths]

    def get_all(self) -> List[Hadith]:
        self._require_loaded()
        return list(self._hadiths)

    def get_metadata(self, ids: List[str]) -> List[Optional[Hadith]]:
        self._require_loaded()
        return [self._id_map.get(hid) for hid in ids]

    def __len__(self) -> int:
        self._require_loaded()
        return len(self._hadiths)
