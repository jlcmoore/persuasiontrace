"""Tests for shared JSONL IO helpers."""

from __future__ import annotations

from pathlib import Path

from simulation.io import read_jsonl_records


def test_read_jsonl_records_ignores_non_dict_and_errors(tmp_path: Path):
    """Default JSONL reader should keep only valid dict rows."""
    jsonl_path = tmp_path / "records.jsonl"
    jsonl_path.write_text(
        "\n".join(
            [
                '{"id": 1}',
                '["not", "a", "dict"]',
                '{"error": "bad"}',
                '{"id": 2}',
            ]
        ),
        encoding="utf-8",
    )

    rows = read_jsonl_records(jsonl_path)
    assert rows == [{"id": 1}, {"id": 2}]


def test_read_jsonl_records_flattens_list_rows_when_enabled(tmp_path: Path):
    """Reader should flatten list payloads when flatten_list_rows is enabled."""
    jsonl_path = tmp_path / "mixed.jsonl"
    jsonl_path.write_text(
        "\n".join(
            [
                '[{"id": "a"}, {"id": "b"}, 3]',
                '{"id": "c"}',
            ]
        ),
        encoding="utf-8",
    )

    rows = read_jsonl_records(
        jsonl_path,
        skip_error_records=False,
        flatten_list_rows=True,
    )
    assert rows == [{"id": "a"}, {"id": "b"}, {"id": "c"}]
