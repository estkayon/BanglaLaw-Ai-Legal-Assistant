import json
from pathlib import Path

import pytest

from src.monitoring.drift import DataDriftMonitor
from src.monitoring.quality import ResponseQualityMonitor


def _write_act(data_dir: Path, act_id: str, num_sections: int = 3, bangla: bool = True) -> None:
    """Helper to create a fake act JSON file."""
    text = "এই আইনে কিছু গুরুত্বপূর্ণ তথ্য রয়েছে।" if bangla else "This act contains important provisions."
    sections = [
        {
            "section_id": str(i),
            "section_number": str(i),
            "section_title": f"Section {i}",
            "section_text": text * 5,
        }
        for i in range(num_sections)
    ]
    act = {"act_id": act_id, "act_title": f"Act {act_id}", "sections": sections}
    filepath = data_dir / f"act_{act_id}.json"
    filepath.write_text(json.dumps(act, ensure_ascii=False), encoding="utf-8")


# --- DataDriftMonitor tests ---


class TestDataDriftMonitor:
    def test_compute_snapshot(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _write_act(data_dir, "1", num_sections=5)
        _write_act(data_dir, "2", num_sections=3)

        monitor = DataDriftMonitor(output_dir=tmp_path)
        snapshot = monitor.compute_snapshot(data_dir)

        assert snapshot["total_acts"] == 2
        assert snapshot["total_sections"] == 8
        assert snapshot["avg_section_length"] > 0
        assert "timestamp" in snapshot

    def test_save_and_load_snapshots(self, tmp_path):
        monitor = DataDriftMonitor(output_dir=tmp_path)
        snapshot = {"timestamp": "2024-01-01", "total_acts": 10, "total_sections": 100}
        monitor.save_snapshot(snapshot)

        loaded = monitor.load_snapshots()
        assert len(loaded) == 1
        assert loaded[0]["total_acts"] == 10

    def test_multiple_snapshots(self, tmp_path):
        monitor = DataDriftMonitor(output_dir=tmp_path)
        monitor.save_snapshot({"timestamp": "2024-01-01", "total_acts": 10, "total_sections": 50})
        monitor.save_snapshot({"timestamp": "2024-02-01", "total_acts": 12, "total_sections": 60})
        loaded = monitor.load_snapshots()
        assert len(loaded) == 2

    def test_no_drift_when_stable(self, tmp_path):
        monitor = DataDriftMonitor(output_dir=tmp_path)
        monitor.save_snapshot({"total_acts": 100, "total_sections": 500, "avg_section_length": 200})

        current = {"total_acts": 102, "total_sections": 510, "avg_section_length": 198}
        warnings = monitor.detect_drift(current)
        assert len(warnings) == 0

    def test_drift_detected(self, tmp_path):
        monitor = DataDriftMonitor(output_dir=tmp_path)
        monitor.save_snapshot({"total_acts": 100, "total_sections": 500, "avg_section_length": 200})

        current = {"total_acts": 150, "total_sections": 500, "avg_section_length": 200}
        warnings = monitor.detect_drift(current)
        assert len(warnings) == 1
        assert "Act count" in warnings[0]

    def test_no_drift_no_history(self, tmp_path):
        monitor = DataDriftMonitor(output_dir=tmp_path)
        current = {"total_acts": 100}
        warnings = monitor.detect_drift(current)
        assert len(warnings) == 0

    def test_run_full(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _write_act(data_dir, "1")

        monitor = DataDriftMonitor(output_dir=tmp_path)
        snapshot, warnings = monitor.run(data_dir)
        assert snapshot["total_acts"] == 1
        assert isinstance(warnings, list)

    def test_empty_data_dir(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        monitor = DataDriftMonitor(output_dir=tmp_path)
        snapshot = monitor.compute_snapshot(data_dir)
        assert snapshot["total_acts"] == 0
        assert snapshot["total_sections"] == 0


# --- ResponseQualityMonitor tests ---


class TestResponseQualityMonitor:
    def test_log_query(self, tmp_path):
        monitor = ResponseQualityMonitor(output_dir=tmp_path)
        monitor.log_query("test question", "en", 0.85, True, 3)

        metrics = monitor.load_metrics()
        assert len(metrics) == 1
        assert metrics[0]["type"] == "query"
        assert metrics[0]["confidence"] == 0.85

    def test_log_feedback(self, tmp_path):
        monitor = ResponseQualityMonitor(output_dir=tmp_path)
        monitor.log_feedback("test question", helpful=True)

        metrics = monitor.load_metrics()
        assert len(metrics) == 1
        assert metrics[0]["type"] == "feedback"
        assert metrics[0]["helpful"] is True

    def test_compute_summary(self, tmp_path):
        monitor = ResponseQualityMonitor(output_dir=tmp_path)

        monitor.log_query("q1", "bn", 0.9, True, 3)
        monitor.log_query("q2", "en", 0.3, False, 0)
        monitor.log_query("q3", "bn", 0.7, True, 2)
        monitor.log_feedback("q1", helpful=True)
        monitor.log_feedback("q2", helpful=False)

        summary = monitor.compute_summary()
        assert summary["total_queries"] == 3
        assert summary["bangla_queries"] == 2
        assert summary["english_queries"] == 1
        assert summary["total_feedback"] == 2
        assert summary["helpful_rate"] == 0.5
        assert summary["confident_rate"] == pytest.approx(0.667, abs=0.01)

    def test_empty_summary(self, tmp_path):
        monitor = ResponseQualityMonitor(output_dir=tmp_path)
        summary = monitor.compute_summary()
        assert summary["total_queries"] == 0
        assert summary["total_feedback"] == 0

    def test_load_empty(self, tmp_path):
        monitor = ResponseQualityMonitor(output_dir=tmp_path)
        assert monitor.load_metrics() == []
