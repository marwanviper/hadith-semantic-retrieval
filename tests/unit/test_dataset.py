"""Unit tests for dataset loading."""

import json
import pytest
from pathlib import Path

from src.datasets.json_loader import JSONHadithDataset
from src.datasets.base import Hadith


@pytest.fixture
def sample_json_file(tmp_path) -> Path:
    data = [
        {"id": "1", "text": "النية من العمل", "book": "البخاري", "narrator": "عمر"},
        {"id": "2", "text": "الدين النصيحة", "book": "مسلم"},
        {"id": "3", "text": "حسن الخلق"},
    ]
    p = tmp_path / "hadiths.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


class TestJSONHadithDataset:
    def test_load_success(self, sample_json_file):
        dataset = JSONHadithDataset(sample_json_file)
        dataset.load()
        assert len(dataset) == 3

    def test_get_texts(self, sample_json_file):
        dataset = JSONHadithDataset(sample_json_file)
        dataset.load()
        texts = dataset.get_texts()
        assert len(texts) == 3
        assert "النية من العمل" in texts

    def test_get_all_returns_hadith_dataclasses(self, sample_json_file):
        dataset = JSONHadithDataset(sample_json_file)
        dataset.load()
        hadiths = dataset.get_all()
        assert all(isinstance(h, Hadith) for h in hadiths)

    def test_optional_fields_default_none(self, sample_json_file):
        dataset = JSONHadithDataset(sample_json_file)
        dataset.load()
        hadiths = dataset.get_all()
        hadith3 = next(h for h in hadiths if h.id == "3")
        assert hadith3.book is None
        assert hadith3.narrator is None

    def test_get_metadata_by_ids(self, sample_json_file):
        dataset = JSONHadithDataset(sample_json_file)
        dataset.load()
        results = dataset.get_metadata(["2", "1"])
        assert results[0].id == "2"
        assert results[1].id == "1"

    def test_missing_id_returns_none(self, sample_json_file):
        dataset = JSONHadithDataset(sample_json_file)
        dataset.load()
        results = dataset.get_metadata(["999"])
        assert results[0] is None

    def test_file_not_found_raises(self, tmp_path):
        dataset = JSONHadithDataset(tmp_path / "missing.json")
        with pytest.raises(FileNotFoundError):
            dataset.load()

    def test_not_loaded_raises(self, sample_json_file):
        dataset = JSONHadithDataset(sample_json_file)
        with pytest.raises(RuntimeError):
            dataset.get_texts()

    def test_jsonl_format(self, tmp_path):
        jsonl_data = '{"id": "1", "text": "حديث أول"}\n{"id": "2", "text": "حديث ثاني"}\n'
        p = tmp_path / "hadiths.jsonl"
        p.write_text(jsonl_data, encoding="utf-8")
        dataset = JSONHadithDataset(p)
        dataset.load()
        assert len(dataset) == 2
