import json

import pytest

from src.ingestion.chunker import LegalChunk, LegalChunker


@pytest.fixture
def chunker():
    return LegalChunker(max_chunk_size=200)


@pytest.fixture
def sample_section():
    return {
        "section_id": "31189",
        "section_number": "২",
        "section_title": "সংজ্ঞা",
        "section_text": "এই আইনে বিষয় বা প্রসঙ্গের পরিপন্থী কিছু না থাকিলে।",
        "section_url": "https://bdlaws.minlaw.gov.bd/act-755/section-31189.html",
    }


@pytest.fixture
def sample_act(sample_section):
    return {
        "act_id": "755",
        "act_title": "মূল্য সংযোজন কর আইন, ১৯৯১",
        "act_number": "২২",
        "sections": [sample_section],
    }


class TestLegalChunk:
    def test_display_source(self):
        chunk = LegalChunk(
            chunk_id="755_s31189",
            act_id="755",
            act_title="মূল্য সংযোজন কর আইন, ১৯৯১",
            section_id="31189",
            section_number="২",
            section_title="সংজ্ঞা",
            text="test",
            metadata={},
        )
        assert "ধারা ২" in chunk.display_source
        assert "মূল্য সংযোজন কর" in chunk.display_source


class TestLegalChunker:
    def test_short_section_single_chunk(self, chunker, sample_section):
        chunks = chunker.chunk_section(sample_section, "755", "Test Act")
        assert len(chunks) == 1
        assert chunks[0].chunk_id == "755_s31189"
        assert chunks[0].section_number == "২"

    def test_empty_section_skipped(self, chunker):
        section = {
            "section_id": "100",
            "section_number": "1",
            "section_title": "Empty",
            "section_text": "",
            "section_url": "",
        }
        chunks = chunker.chunk_section(section, "1", "Act")
        assert len(chunks) == 0

    def test_large_section_split(self):
        chunker = LegalChunker(max_chunk_size=100)
        long_text = (
            "(ক) প্রথম উপধারা এখানে লেখা আছে যা অনেক বড়।\n"
            "(খ) দ্বিতীয় উপধারা এখানে আছে এটাও বড়।\n"
            "(গ) তৃতীয় উপধারা এটা আরও বড় লেখা।"
        )
        section = {
            "section_id": "500",
            "section_number": "৫",
            "section_title": "বড় ধারা",
            "section_text": long_text,
            "section_url": "",
        }
        chunks = chunker.chunk_section(section, "10", "Test")
        assert len(chunks) >= 2
        # Part index reflected in chunk_id
        assert any("_p" in c.chunk_id for c in chunks[1:])

    def test_chunk_act(self, chunker, sample_act):
        chunks = chunker.chunk_act(sample_act)
        assert len(chunks) == 1
        assert chunks[0].act_id == "755"
        assert chunks[0].act_title == "মূল্য সংযোজন কর আইন, ১৯৯১"

    def test_chunk_act_multiple_sections(self, chunker):
        act = {
            "act_id": "10",
            "act_title": "Test Act",
            "sections": [
                {
                    "section_id": "100",
                    "section_number": "১",
                    "section_title": "শিরোনাম",
                    "section_text": "এই আইন টেস্ট আইন।",
                    "section_url": "",
                },
                {
                    "section_id": "101",
                    "section_number": "২",
                    "section_title": "সংজ্ঞা",
                    "section_text": "এই আইনে কিছু সংজ্ঞা আছে।",
                    "section_url": "",
                },
            ],
        }
        chunks = chunker.chunk_act(act)
        assert len(chunks) == 2
        assert chunks[0].section_number == "১"
        assert chunks[1].section_number == "২"

    def test_chunk_from_file(self, chunker, sample_act, tmp_path):
        filepath = tmp_path / "act_755.json"
        filepath.write_text(json.dumps(sample_act, ensure_ascii=False), encoding="utf-8")

        chunks = chunker.chunk_from_file(filepath)
        assert len(chunks) == 1

    def test_chunk_all_files(self, chunker, sample_act, tmp_path):
        for i in range(3):
            act = {**sample_act, "act_id": str(i)}
            filepath = tmp_path / f"act_{i}.json"
            filepath.write_text(json.dumps(act, ensure_ascii=False), encoding="utf-8")

        chunks = chunker.chunk_all_files(tmp_path)
        assert len(chunks) == 3

    def test_chunk_all_files_empty_dir(self, chunker, tmp_path):
        chunks = chunker.chunk_all_files(tmp_path)
        assert len(chunks) == 0

    def test_metadata_preserved(self, chunker, sample_section):
        chunks = chunker.chunk_section(sample_section, "755", "Test Act")
        meta = chunks[0].metadata
        assert meta["act_id"] == "755"
        assert meta["section_id"] == "31189"
        assert meta["source_url"] == sample_section["section_url"]
