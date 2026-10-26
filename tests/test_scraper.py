import json
from pathlib import Path

from src.scraper import BDLawsScraper, ChangeDetector, ChangeResult, LawAct, LawSection

# --- BDLawsScraper tests ---


class TestBDLawsScraper:
    def test_law_section_dataclass(self):
        section = LawSection(
            section_id="31189",
            section_number="২",
            section_title="সংজ্ঞা",
            section_text="এই আইনে...",
            section_url="https://bdlaws.minlaw.gov.bd/act-755/section-31189.html",
        )
        assert section.section_id == "31189"
        assert section.section_number == "২"

    def test_law_act_dataclass_defaults(self):
        act = LawAct(
            act_id="755",
            act_title="মূল্য সংযোজন কর আইন, ১৯৯১",
            act_number="২২",
            act_year="১৯৯১",
            act_date="৩০ জুন, ১৯৯১",
            act_url="https://bdlaws.minlaw.gov.bd/act-755.html",
            volume="29",
        )
        assert act.act_id == "755"
        assert act.sections == []
        assert act.preamble == ""

    def test_save_act_creates_json(self, tmp_path):
        scraper = BDLawsScraper(output_dir=tmp_path)
        act = LawAct(
            act_id="100",
            act_title="Test Act",
            act_number="1",
            act_year="2024",
            act_date="1 Jan 2024",
            act_url="https://bdlaws.minlaw.gov.bd/act-100.html",
            volume="1",
            preamble="Test preamble",
            sections=[
                LawSection(
                    section_id="5000",
                    section_number="1",
                    section_title="Short title",
                    section_text="This act may be called the Test Act.",
                    section_url="https://bdlaws.minlaw.gov.bd/act-100/section-5000.html",
                )
            ],
        )

        filepath = scraper.save_act(act)
        assert filepath.exists()
        assert filepath.name == "act_100.json"

        data = json.loads(filepath.read_text(encoding="utf-8"))
        assert data["act_id"] == "100"
        assert data["act_title"] == "Test Act"
        assert len(data["sections"]) == 1
        assert data["sections"][0]["section_text"] == "This act may be called the Test Act."
        scraper.close()

    def test_save_act_preserves_bangla(self, tmp_path):
        scraper = BDLawsScraper(output_dir=tmp_path)
        act = LawAct(
            act_id="755",
            act_title="মূল্য সংযোজন কর আইন, ১৯৯১",
            act_number="২২",
            act_year="১৯৯১",
            act_date="",
            act_url="https://bdlaws.minlaw.gov.bd/act-755.html",
            volume="29",
        )

        filepath = scraper.save_act(act)
        data = json.loads(filepath.read_text(encoding="utf-8"))
        assert "মূল্য সংযোজন কর" in data["act_title"]
        scraper.close()

    def test_compute_hash_consistent(self, tmp_path):
        scraper = BDLawsScraper(output_dir=tmp_path)
        test_file = tmp_path / "test.json"
        test_file.write_text('{"key": "value"}', encoding="utf-8")

        hash1 = scraper.compute_hash(test_file)
        hash2 = scraper.compute_hash(test_file)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex
        scraper.close()

    def test_output_dir_created(self, tmp_path):
        output = tmp_path / "nested" / "output"
        scraper = BDLawsScraper(output_dir=output)
        assert output.exists()
        scraper.close()


# --- ChangeDetector tests ---


class TestChangeDetector:
    def _write_act(self, data_dir: Path, act_id: str, content: str) -> Path:
        filepath = data_dir / f"act_{act_id}.json"
        filepath.write_text(content, encoding="utf-8")
        return filepath

    def test_all_new_files(self, tmp_path):
        self._write_act(tmp_path, "1", '{"title": "Act 1"}')
        self._write_act(tmp_path, "2", '{"title": "Act 2"}')

        detector = ChangeDetector(data_dir=tmp_path)
        result = detector.detect_changes()

        assert len(result.new_files) == 2
        assert len(result.modified_files) == 0
        assert result.has_changes is True

    def test_unchanged_after_hash_update(self, tmp_path):
        self._write_act(tmp_path, "1", '{"title": "Act 1"}')

        detector = ChangeDetector(data_dir=tmp_path)
        detector.update_hashes()

        result = detector.detect_changes()
        assert len(result.unchanged_files) == 1
        assert len(result.new_files) == 0
        assert result.has_changes is False

    def test_modified_file_detected(self, tmp_path):
        filepath = self._write_act(tmp_path, "1", '{"title": "Act 1"}')

        detector = ChangeDetector(data_dir=tmp_path)
        detector.update_hashes()

        # Modify file
        filepath.write_text('{"title": "Act 1 Updated"}', encoding="utf-8")

        result = detector.detect_changes()
        assert len(result.modified_files) == 1
        assert result.has_changes is True

    def test_get_changed_files_updates_hashes(self, tmp_path):
        self._write_act(tmp_path, "1", '{"title": "Act 1"}')

        detector = ChangeDetector(data_dir=tmp_path)
        changed = detector.get_changed_files()
        assert len(changed) == 1

        # Second call should find no changes
        changed_again = detector.get_changed_files()
        assert len(changed_again) == 0

    def test_change_result_summary(self):
        result = ChangeResult(
            new_files=[Path("a.json")],
            modified_files=[Path("b.json"), Path("c.json")],
            unchanged_files=[Path("d.json")],
        )
        assert "New: 1" in result.summary
        assert "Modified: 2" in result.summary
        assert "Unchanged: 1" in result.summary

    def test_corrupted_hash_file_handled(self, tmp_path):
        hash_file = tmp_path / "file_hashes.json"
        hash_file.write_text("not valid json", encoding="utf-8")

        detector = ChangeDetector(data_dir=tmp_path)
        assert detector.hashes == {}
