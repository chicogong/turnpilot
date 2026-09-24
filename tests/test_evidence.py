from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from turnpilot.evidence import audit_evidence_manifest, main

EXAMPLE = Path(__file__).parents[1] / "examples" / "synthetic-evidence.json"


def example() -> dict[str, object]:
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def test_audit_counts_readiness_without_emitting_identifiers() -> None:
    report = audit_evidence_manifest(example())
    assert report["sessions"] == 1
    assert report["candidates"] == 2
    assert report["adjudicated_candidates"] == 2
    assert report["action_ready_candidates"] == 2
    assert report["cutoff_ready_candidates"] == 2
    assert report["ambiguous_candidates"] == 1
    assert report["asr_revision_events"] == 3
    assert report["sessions_with_recorded_clock_and_all_capture"] == 0
    assert report["heldout_isolated_by_declared_ids"] is None
    output = json.dumps(report)
    assert "synthetic-session" not in output
    assert "synthetic-speaker" not in output
    assert "synthetic-complete" not in output


def test_audit_rejects_raw_transcript_and_audio_fields() -> None:
    manifest = example()
    session = manifest["sessions"][0]
    session["transcript"] = "private words"
    with pytest.raises(ValueError, match="invalid session object"):
        audit_evidence_manifest(manifest)
    del session["transcript"]
    session["candidates"][0]["audio_path"] = "/private/audio.wav"
    with pytest.raises(ValueError, match="invalid candidate object"):
        audit_evidence_manifest(manifest)


def test_audit_rejects_noncausal_asr_and_playback() -> None:
    manifest = example()
    session = manifest["sessions"][0]
    session["asr_revisions"][1]["available_at_ms"] = 400
    with pytest.raises(ValueError, match="ASR revisions"):
        audit_evidence_manifest(manifest)
    session["asr_revisions"][1]["available_at_ms"] = 1400
    session["playback_intervals"][0]["start_ms"] = -1
    with pytest.raises(ValueError, match="playback.start_ms"):
        audit_evidence_manifest(manifest)


def test_audit_rejects_capture_gaps_and_overlapping_playback() -> None:
    manifest = example()
    session = manifest["sessions"][0]
    session["capture"]["asr_revisions"] = False
    with pytest.raises(ValueError, match="ASR events require"):
        audit_evidence_manifest(manifest)
    session["capture"]["asr_revisions"] = True
    session["capture"]["playback"] = False
    with pytest.raises(ValueError, match="playback intervals require"):
        audit_evidence_manifest(manifest)
    session["capture"]["playback"] = True
    session["playback_intervals"].append({"start_ms": 900, "end_ms": 1000})
    with pytest.raises(ValueError, match="non-overlapping"):
        audit_evidence_manifest(manifest)


def test_audit_rejects_inconsistent_label_and_review() -> None:
    manifest = example()
    label = manifest["sessions"][0]["candidates"][1]["label"]
    label["is_complete"] = True
    with pytest.raises(ValueError, match="continuation"):
        audit_evidence_manifest(manifest)
    label["is_complete"] = False
    label["reviewer_count"] = 1
    with pytest.raises(ValueError, match="review count"):
        audit_evidence_manifest(manifest)


def test_audit_rejects_non_user_completion_and_invalid_action() -> None:
    manifest = example()
    label = manifest["sessions"][0]["candidates"][0]["label"]
    label["origin"] = "echo"
    with pytest.raises(ValueError, match="non-user candidate"):
        audit_evidence_manifest(manifest)
    label["origin"] = "near_end_user"
    label["expected_action"] = "speak_now"
    with pytest.raises(ValueError, match="expected_action"):
        audit_evidence_manifest(manifest)
    label["expected_action"] = "commit_user_turn"
    label["true_eot_ms"] = 400
    with pytest.raises(ValueError, match="true_eot"):
        audit_evidence_manifest(manifest)


def test_audit_rejects_bad_candidate_intervals_and_duplicate_ids() -> None:
    manifest = example()
    first = manifest["sessions"][0]["candidates"][0]
    first["label"]["speech_end_ms"] = None
    with pytest.raises(ValueError, match="speech interval"):
        audit_evidence_manifest(manifest)
    first["label"]["speech_end_ms"] = 500
    first["label"]["quality"] = "not_speech"
    with pytest.raises(ValueError, match="not_speech"):
        audit_evidence_manifest(manifest)
    first["label"]["quality"] = "clear"
    manifest["sessions"][0]["candidates"][1]["candidate_id"] = first["candidate_id"]
    with pytest.raises(ValueError, match="candidate IDs"):
        audit_evidence_manifest(manifest)


def test_audit_rejects_invalid_root_session_and_boolean_values() -> None:
    manifest = example()
    manifest["schema_version"] = 2
    with pytest.raises(ValueError, match="unsupported evidence schema"):
        audit_evidence_manifest(manifest)
    manifest["schema_version"] = 1
    manifest["evidence_level"] = "device_proven"
    with pytest.raises(ValueError, match="evidence_level"):
        audit_evidence_manifest(manifest)
    manifest["evidence_level"] = "synthetic"
    manifest["evidence_level"] = "human_reviewed"
    with pytest.raises(ValueError, match="synthetic source"):
        audit_evidence_manifest(manifest)
    manifest["evidence_level"] = "synthetic"
    manifest["sessions"][0]["audio_sha256"] = "not-a-hash"
    with pytest.raises(ValueError, match="audio_sha256"):
        audit_evidence_manifest(manifest)
    manifest["sessions"][0]["audio_sha256"] = "0" * 64
    manifest["sessions"][0]["capture"]["playback"] = 1
    with pytest.raises(ValueError, match="boolean"):
        audit_evidence_manifest(manifest)


def test_unreviewed_and_missing_capture_are_not_counted_as_ready() -> None:
    manifest = example()
    session = manifest["sessions"][0]
    session["capture"]["asr_revisions"] = False
    session["asr_revisions"] = []
    label = session["candidates"][0]["label"]
    label["adjudicated"] = False
    label["reviewer_count"] = 0
    report = audit_evidence_manifest(manifest)
    assert report["adjudicated_candidates"] == 1
    assert report["action_ready_candidates"] == 1
    assert report["asr_revision_events"] == 0


def test_reviewed_noise_is_action_ready_but_not_cutoff_ready() -> None:
    manifest = example()
    label = manifest["sessions"][0]["candidates"][0]["label"]
    label.update(
        {
            "speech_start_ms": None,
            "speech_end_ms": None,
            "origin": "noise",
            "quality": "not_speech",
            "is_complete": None,
            "true_eot_ms": None,
            "expected_action": "ignore_user_turn",
        }
    )
    report = audit_evidence_manifest(manifest)
    assert report["action_ready_candidates"] == 2
    assert report["cutoff_ready_candidates"] == 1
    assert report["origin_candidates"]["noise"] == 1


def test_split_overlap_is_reported_without_speaker_or_device_id() -> None:
    manifest = example()
    heldout = deepcopy(manifest["sessions"][0])
    heldout["session_id"] = "heldout-secret-session"
    heldout["split"] = "heldout"
    manifest["sessions"].append(heldout)
    report = audit_evidence_manifest(manifest)
    assert report["heldout_isolated_by_declared_ids"] is False
    assert report["cross_split_speaker_overlap"] == 1
    assert report["cross_split_device_overlap"] == 1
    assert "heldout-secret-session" not in json.dumps(report)
    heldout["speaker_id"] = "new-speaker"
    heldout["device_id"] = "new-device"
    assert audit_evidence_manifest(manifest)["heldout_isolated_by_declared_ids"] is True


def test_cli_outputs_aggregate_and_hides_bad_input(tmp_path, capsys) -> None:
    assert main([str(EXAMPLE)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["candidates"] == 2
    path = tmp_path / "private.json"
    path.write_text('{"transcript":"private words"}', encoding="utf-8")
    assert main([str(path)]) == 2
    output = capsys.readouterr()
    assert "private words" not in output.err + output.out
