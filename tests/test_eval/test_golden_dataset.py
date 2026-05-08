"""Tests for the golden dataset structure and integrity."""

import json
from pathlib import Path

import pytest

DATASET_PATH = Path(__file__).parent.parent.parent / "eval" / "golden_dataset.json"

VALID_TOOLS = {"sql_query", "doc_search", "calculator"}
VALID_CATEGORIES = {
    "single-tool-sql",
    "single-tool-rag",
    "single-tool-calculator",
    "multi-tool",
}


@pytest.fixture
def dataset() -> list[dict]:
    with open(DATASET_PATH) as f:
        return json.load(f)


class TestDatasetStructure:
    def test_dataset_is_list(self, dataset: list[dict]) -> None:
        assert isinstance(dataset, list)

    def test_dataset_has_15_entries(self, dataset: list[dict]) -> None:
        assert len(dataset) == 15

    def test_all_entries_have_required_fields(self, dataset: list[dict]) -> None:
        required = {"id", "query", "expected_tools", "expected_answer_contains", "category"}
        for entry in dataset:
            missing = required - set(entry.keys())
            assert not missing, f"Entry {entry.get('id', '?')} missing fields: {missing}"

    def test_ids_are_unique(self, dataset: list[dict]) -> None:
        ids = [e["id"] for e in dataset]
        assert len(ids) == len(set(ids)), f"Duplicate IDs found: {ids}"

    def test_all_tools_are_valid(self, dataset: list[dict]) -> None:
        for entry in dataset:
            for tool in entry["expected_tools"]:
                assert tool in VALID_TOOLS, (
                    f"Entry {entry['id']} has invalid tool: {tool}"
                )

    def test_all_categories_are_valid(self, dataset: list[dict]) -> None:
        for entry in dataset:
            assert entry["category"] in VALID_CATEGORIES, (
                f"Entry {entry['id']} has invalid category: {entry['category']}"
            )


class TestDatasetContent:
    def test_single_tool_entries_have_one_tool(self, dataset: list[dict]) -> None:
        for entry in dataset:
            if entry["category"].startswith("single-tool"):
                assert len(entry["expected_tools"]) == 1, (
                    f"Single-tool entry {entry['id']} has {len(entry['expected_tools'])} tools"
                )

    def test_multi_tool_entries_have_multiple_tools(self, dataset: list[dict]) -> None:
        for entry in dataset:
            if entry["category"] == "multi-tool":
                assert len(entry["expected_tools"]) >= 2, (
                    f"Multi-tool entry {entry['id']} has only {len(entry['expected_tools'])} tool(s)"
                )

    def test_expected_answer_contains_not_empty(self, dataset: list[dict]) -> None:
        for entry in dataset:
            assert len(entry["expected_answer_contains"]) >= 1, (
                f"Entry {entry['id']} has no expected keywords"
            )

    def test_queries_are_non_empty_strings(self, dataset: list[dict]) -> None:
        for entry in dataset:
            assert isinstance(entry["query"], str)
            assert len(entry["query"].strip()) > 0, f"Entry {entry['id']} has empty query"

    def test_category_counts(self, dataset: list[dict]) -> None:
        counts: dict[str, int] = {}
        for entry in dataset:
            counts[entry["category"]] = counts.get(entry["category"], 0) + 1
        assert counts.get("single-tool-sql", 0) == 5
        assert counts.get("single-tool-rag", 0) == 3
        assert counts.get("single-tool-calculator", 0) == 2
        assert counts.get("multi-tool", 0) == 5
