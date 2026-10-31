"""
Abstract base class for Hadith dataset loaders.

Design intent:
  The search engine should never know whether data came from JSON, SQLite,
  a REST API, or an Android asset bundle.  Any new data source just needs
  to implement this interface.

Android portability:
  At inference time on Android, the dataset is NOT loaded — only the
  pre-built index and metadata mapping (id → text/metadata) are needed.
  This class is an offline-only concern.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class Hadith:
    """A single Hadith record."""
    id: str
    text: str
    book: Optional[str] = None
    chapter: Optional[str] = None
    narrator: Optional[str] = None
    source: Optional[str] = None
    reference: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "book": self.book,
            "chapter": self.chapter,
            "narrator": self.narrator,
            "source": self.source,
            "reference": self.reference,
        }


class HadithDataset(ABC):
    """
    Abstract Hadith dataset loader.

    Subclass this to add new data sources (JSON, SQLite, XLSX, …).
    """

    @abstractmethod
    def load(self) -> None:
        """
        Load the dataset into memory.

        Must be called before any other method.  Implementations should
        log progress and raise descriptive exceptions on failure.
        """
        ...

    @abstractmethod
    def get_texts(self) -> List[str]:
        """
        Return the text of every Hadith in load order.

        Returns:
            List of raw text strings, one per Hadith.
        """
        ...

    @abstractmethod
    def get_all(self) -> List[Hadith]:
        """
        Return all Hadith records.

        Returns:
            List of Hadith dataclass instances.
        """
        ...

    @abstractmethod
    def get_metadata(self, ids: List[str]) -> List[Hadith]:
        """
        Return Hadith records for the given IDs.

        Args:
            ids: List of Hadith ID strings.

        Returns:
            List of Hadith records in the same order as ids.
            Missing IDs return None in the corresponding position.
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of Hadiths."""
        ...
