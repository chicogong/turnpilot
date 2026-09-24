from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SOURCE = Path(__file__).resolve().parents[1] / "examples" / "easyturn_jev_probe.py"
_SPEC = importlib.util.spec_from_file_location("easyturn_jev_probe_for_test", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
probe = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = probe
_SPEC.loader.exec_module(probe)


def _item(label: str, origin: str, number: int) -> str:
    return json.dumps(
        {
            "key": f"{label}_{origin}_{number}",
            "txt": f"测试内容{number}<{label.upper()}>",
            "lang": "<CN>",
            "speaker": f"speaker-{number}",
            "extra": {"dataset": f"{label}_test_{origin}"},
        },
        ensure_ascii=False,
    )


def test_public_list_load_and_stratified_sample(tmp_path: Path) -> None:
    complete_path = tmp_path / "complete.list"
    incomplete_path = tmp_path / "incomplete.list"
    complete_path.write_text(
        "\n".join(
            _item("complete", origin, number)
            for origin in ("real", "synthetic")
            for number in range(3)
        ),
        encoding="utf-8",
    )
    incomplete_path.write_text(
        "\n".join(
            _item("incomplete", origin, number)
            for origin in ("real", "synthetic")
            for number in range(3)
        ),
        encoding="utf-8",
    )
    rows = probe._load(complete_path, label="complete") + probe._load(
        incomplete_path, label="incomplete"
    )
    selected = probe._sample(rows, 2)
    assert selected == probe._sample(rows, 2)
    assert len(selected) == 8
    real_only = probe._sample(rows, 2, ("real",))
    assert len(real_only) == 4
    assert all(row["origin"] == "real" for row in real_only)
    assert all("<COMPLETE>" not in row["text"] for row in selected)
    assert all("<INCOMPLETE>" not in row["text"] for row in selected)
    assert {
        (label, origin): sum(row["label"] == label and row["origin"] == origin for row in selected)
        for label in ("complete", "incomplete")
        for origin in ("real", "synthetic")
    } == {
        ("complete", "real"): 2,
        ("complete", "synthetic"): 2,
        ("incomplete", "real"): 2,
        ("incomplete", "synthetic"): 2,
    }


def test_public_list_rejects_wrong_label(tmp_path: Path) -> None:
    path = tmp_path / "wrong.list"
    path.write_text(_item("incomplete", "real", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected Easy Turn label"):
        probe._load(path, label="complete")


def test_percentile_nearest_rank() -> None:
    assert probe._percentile([], 50) is None
    assert probe._percentile([100, 200, 300, 400], 50) == 200
    assert probe._percentile([100, 200, 300, 400], 95) == 400
