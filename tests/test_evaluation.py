import json
from pathlib import Path

import pytest

from src.evaluation.metrics import EvalMetrics, EvalSummary, compute_summary


@pytest.fixture
def high_confidence_result():
    return EvalMetrics(
        qa_id="qa_001",
        question="ভাড়াটিয়া উচ্ছেদের আইন কি?",
        expected_act="প্রাঙ্গণ ভাড়া নিয়ন্ত্রণ আইন, ১৯৯১",
        expected_section="১২",
        predicted_answer="প্রাঙ্গণ ভাড়া নিয়ন্ত্রণ আইন অনুযায়ী ভাড়াটিয়াকে উচ্ছেদের জন্য নোটিশ দিতে হয়।",
        predicted_citations=[
            {
                "act_title": "প্রাঙ্গণ ভাড়া নিয়ন্ত্রণ আইন, ১৯৯১",
                "section_number": "১২",
                "source": "প্রাঙ্গণ ভাড়া নিয়ন্ত্রণ আইন, ধারা ১২",
                "url": "",
                "confidence": 0.85,
            }
        ],
        confidence=0.85,
        is_confident=True,
    )


@pytest.fixture
def low_confidence_result():
    return EvalMetrics(
        qa_id="qa_002",
        question="test question",
        expected_act="Test Act",
        expected_section="5",
        predicted_answer="Not enough info.",
        predicted_citations=[],
        confidence=0.2,
        is_confident=False,
    )


@pytest.fixture
def wrong_citation_result():
    return EvalMetrics(
        qa_id="qa_003",
        question="test question",
        expected_act="Expected Act",
        expected_section="10",
        predicted_answer="Some long answer that has more than twenty characters for sure.",
        predicted_citations=[
            {
                "act_title": "Wrong Act",
                "section_number": "99",
                "source": "Wrong",
                "url": "",
                "confidence": 0.7,
            }
        ],
        confidence=0.7,
        is_confident=True,
    )


class TestEvalMetrics:
    def test_citation_hit_match(self, high_confidence_result):
        assert high_confidence_result.citation_hit is True

    def test_citation_hit_no_match(self, wrong_citation_result):
        assert wrong_citation_result.citation_hit is False

    def test_section_hit_match(self, high_confidence_result):
        assert high_confidence_result.section_hit is True

    def test_section_hit_no_match(self, wrong_citation_result):
        assert wrong_citation_result.section_hit is False

    def test_has_answer_confident(self, high_confidence_result):
        assert high_confidence_result.has_answer is True

    def test_has_answer_low_confidence(self, low_confidence_result):
        assert low_confidence_result.has_answer is False

    def test_to_dict(self, high_confidence_result):
        d = high_confidence_result.to_dict()
        assert d["qa_id"] == "qa_001"
        assert d["citation_hit"] is True
        assert d["section_hit"] is True
        assert "confidence" in d


class TestEvalSummary:
    def test_rates_calculated(self):
        summary = EvalSummary(
            total=10,
            answered=8,
            citation_hits=6,
            section_hits=4,
            avg_confidence=0.75,
        )
        assert summary.answer_rate == pytest.approx(0.8)
        assert summary.citation_accuracy == pytest.approx(0.6)
        assert summary.section_accuracy == pytest.approx(0.4)

    def test_zero_total(self):
        summary = EvalSummary(0, 0, 0, 0, 0.0)
        assert summary.answer_rate == 0.0
        assert summary.citation_accuracy == 0.0

    def test_to_dict(self):
        summary = EvalSummary(5, 4, 3, 2, 0.65)
        d = summary.to_dict()
        assert d["total"] == 5
        assert "answer_rate" in d
        assert "citation_accuracy" in d


class TestComputeSummary:
    def test_aggregation(self, high_confidence_result, low_confidence_result):
        results = [high_confidence_result, low_confidence_result]
        summary = compute_summary(results)
        assert summary.total == 2
        assert summary.answered == 1
        assert summary.citation_hits == 1

    def test_empty_results(self):
        summary = compute_summary([])
        assert summary.total == 0
        assert summary.avg_confidence == 0.0

    def test_all_correct(self, high_confidence_result):
        results = [high_confidence_result] * 3
        summary = compute_summary(results)
        assert summary.answer_rate == 1.0
        assert summary.citation_accuracy == 1.0


class TestLoadQAPairs:
    def test_load_from_file(self, tmp_path):
        from src.evaluation.evaluate import load_qa_pairs

        qa_data = [{"id": "qa_001", "question": "test?"}]
        filepath = tmp_path / "test_qa.json"
        filepath.write_text(json.dumps(qa_data), encoding="utf-8")

        loaded = load_qa_pairs(filepath)
        assert len(loaded) == 1
        assert loaded[0]["id"] == "qa_001"

    def test_file_not_found(self):
        from src.evaluation.evaluate import load_qa_pairs

        with pytest.raises(FileNotFoundError):
            load_qa_pairs(Path("nonexistent.json"))
