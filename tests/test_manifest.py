from __future__ import annotations

import json

import pytest

from turnpilot.manifest import evaluate_manifest, main


def synthetic_manifest() -> dict[str, object]:
    acoustic = {
        "observed_at_ms": 1000,
        "speech_active": False,
        "pause_duration_ms": 640,
        "audio_quality": "clear",
    }
    semantic = {
        "received_at_ms": 990,
        "transcript_revision": 1,
        "complete_probability": 0.9,
        "clarification_probability": 0.0,
        "backchannel_probability": 0.0,
        "response_probability": 0.9,
        "model": "synthetic",
    }
    return {
        "schema_version": 1,
        "evidence_level": "synthetic",
        "cases": [
            {
                "case_id": "synthetic-1",
                "session_id": "s",
                "turn_id": "t",
                "generation": 0,
                "now_ms": 1000,
                "host": {"session_active": True},
                "transcript": {"available_at_ms": 900, "revision": 1, "has_text": True},
                "label": {"is_complete": False, "expected_action": "wait"},
                "arms": {
                    "A": {"acoustic": acoustic},
                    "B": {"acoustic": {**acoustic, "pause_duration_ms": 320}},
                    "C": {"acoustic": acoustic, "semantic": semantic},
                    "D": {
                        "acoustic": {**acoustic, "pause_duration_ms": 96},
                        "semantic": semantic,
                    },
                },
            }
        ],
    }


def test_manifest_runs_four_arms_without_text_output() -> None:
    result = evaluate_manifest(synthetic_manifest())
    assert result["declared_evidence_level"] == "synthetic"
    arms = result["arms"]
    assert isinstance(arms, dict)
    assert [arms[name]["false_cutoffs"] for name in ("A", "B", "C", "D")] == [1, 0, 1, 0]
    assert arms["A"]["action_confusion"] == [
        {"expected": "wait", "actual": "commit_user_turn", "count": 1}
    ]
    assert result["paired_false_cutoff_vs_A"]["B"] is None
    assert "synthetic-1" not in json.dumps(result)


def test_manifest_reports_paired_interval_with_speaker_metadata() -> None:
    from copy import deepcopy

    document = synthetic_manifest()
    first = document["cases"][0]
    rows = []
    for index in range(10):
        row = deepcopy(first)
        row["case_id"] = f"candidate-{index}"
        row["turn_id"] = f"turn-{index}"
        row["now_ms"] = 1000 + index
        row["label"]["speaker_id"] = f"speaker-{index}"
        row["label"]["device_id"] = f"device-{index}"
        for arm in ("A", "B", "C", "D"):
            row["arms"][arm]["acoustic"]["observed_at_ms"] = 1000 + index
            if "semantic" in row["arms"][arm]:
                row["arms"][arm]["semantic"]["received_at_ms"] = 990 + index
        row["transcript"]["available_at_ms"] = 900 + index
        rows.append(row)
    document["cases"] = rows
    result = evaluate_manifest(document)
    interval = result["paired_false_cutoff_vs_A"]["B"]
    assert interval["rate_delta"] == -1.0
    assert interval["ci95_lower"] == -1.0
    assert interval["ci95_upper"] == -1.0
    assert interval["speaker_clusters"] == 10
    assert "speaker-0" not in json.dumps(result)


def test_manifest_rejects_private_transcript_text() -> None:
    document = synthetic_manifest()
    cases = document["cases"]
    assert isinstance(cases, list)
    transcript = cases[0]["transcript"]
    transcript["text"] = "private text"
    with pytest.raises(ValueError, match="must not contain transcript text"):
        evaluate_manifest(document)


def test_cli_errors_never_print_private_input(tmp_path, capsys) -> None:
    path = tmp_path / "private.json"
    path.write_text('{"secret":"private transcript"}', encoding="utf-8")
    assert main([str(path)]) == 2
    output = capsys.readouterr()
    assert "private transcript" not in output.err
    assert "private transcript" not in output.out


def test_cli_emits_aggregate_json(tmp_path, capsys) -> None:
    path = tmp_path / "synthetic.json"
    path.write_text(json.dumps(synthetic_manifest()), encoding="utf-8")
    assert main([str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["arms"]["A"]["false_cutoff_rate"] == 1.0
    assert "synthetic-1" not in json.dumps(result)
