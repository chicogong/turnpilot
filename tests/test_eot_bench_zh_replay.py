from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SOURCE = Path(__file__).resolve().parents[1] / "examples" / "eot_bench_zh_replay.py"
_SPEC = importlib.util.spec_from_file_location("eot_bench_zh_replay_for_test", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
replay = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = replay
_SPEC.loader.exec_module(replay)


def test_label_alignment_is_post_hoc_and_resume_wins_at_boundary() -> None:
    spans = replay.eligible_spans(
        [
            {"start": 1.0, "end": 1.8},
            {"start": 2.0, "end": 2.8},
        ],
        audio_duration_ms=2800,
    )
    assert len(spans) == 2
    assert spans[0].is_hold and not spans[1].is_hold
    assert replay.endpoint_in_span((1800, 2800), spans[0]) is None
    assert replay.endpoint_in_span((1800, 2800), spans[1]) == 2800
    assert replay.endpoint_in_span((1640, 2800), spans[0]) == 1640
    counts = replay.ArmCounts()
    replay.add_span_outcome(counts, spans[0], 1640)
    replay.add_span_outcome(counts, spans[1], 2800)
    assert counts.hold_cutoffs == 1
    assert counts.hold_cutoffs_in_short_holds == 0
    assert counts.eot_detected_within_recording == 1
    assert counts.eot_detected_in_short_eot == 0
    assert counts.eot_latencies_ms == [800]


def test_short_labeled_span_can_contain_endpoint_from_earlier_gate_pause() -> None:
    hold = replay.LabeledSpan(1000, 1300, True)
    eot = replay.LabeledSpan(1400, 1700, False)
    counts = replay.ArmCounts()
    replay.add_span_outcome(counts, hold, 1100)
    replay.add_span_outcome(counts, eot, 1500)
    assert counts.hold_cutoffs_in_short_holds == 1
    assert counts.eot_detected_in_short_eot == 1
    assert counts.eot_latencies_ms == [100]


def test_final_publisher_span_remains_eot_after_filtering() -> None:
    spans = replay.eligible_spans(
        [
            {"start": 0.0, "end": 0.05},
            {"start": 0.1, "end": 5.2},
            {"start": 5.3, "end": 5.9},
        ],
        audio_duration_ms=5900,
    )
    assert spans == (replay.LabeledSpan(5300, 5900, False, 5200),)


def test_preceding_speech_window_uses_previous_raw_span_even_when_filtered() -> None:
    spans = replay.eligible_spans(
        [
            {"start": 0.2, "end": 0.25},
            {"start": 0.4, "end": 1.04},
            {"start": 1.2, "end": 1.84},
        ],
        audio_duration_ms=1840,
    )
    assert spans == (
        replay.LabeledSpan(400, 1040, True, 250),
        replay.LabeledSpan(1200, 1840, False, 1040),
    )
    first = replay.AcousticEndpointTrace((), (224, 256, 384, 1056))
    second = replay.AcousticEndpointTrace((), (224, 1040, 1184))
    assert replay.preceding_speech_observed(spans[0], first)
    assert not replay.preceding_speech_observed(spans[0], second)
    assert replay.preceding_speech_observed(spans[1], first)
    assert replay.preceding_speech_observed(spans[1], second)


def test_silence_boundary_or_adjacent_spans_do_not_fake_preceding_speech() -> None:
    spans = replay.eligible_spans(
        [{"start": 0.4, "end": 1.04}, {"start": 1.04, "end": 1.84}],
        audio_duration_ms=1840,
    )
    first = replay.AcousticEndpointTrace((), (400,))
    second = replay.AcousticEndpointTrace((), (1040,))
    assert not replay.preceding_speech_observed(spans[0], first)
    assert not replay.preceding_speech_observed(spans[1], second)


def test_frame_coverage_requires_pre_resume_for_hold_but_allows_eof_for_eot() -> None:
    frame_times = (1640,)
    hold = replay.LabeledSpan(1000, 1640, True)
    eot = replay.LabeledSpan(1000, 1640, False)
    assert not replay.frame_covers_label_deadline(hold, frame_times)
    assert replay.frame_covers_label_deadline(eot, frame_times)


def test_long_observed_and_jointly_evaluable_denominators() -> None:
    spans = (
        replay.LabeledSpan(400, 900, True, 0),
        replay.LabeledSpan(1200, 1900, True, 900),
        replay.LabeledSpan(2000, 2800, False, 1900),
    )
    fixed = replay.AcousticEndpointTrace((1500, 2640), (384, 1184, 1984))
    dynamic = replay.AcousticEndpointTrace((2640,), (384, 1984))
    fixed_counts = replay.ArmCounts()
    dynamic_counts = replay.ArmCounts()
    paired = replay._empty_paired_evaluable()
    frame_times = (1840, 2640)
    for span in spans:
        replay.add_evaluable_outcome(fixed_counts, span, fixed)
        replay.add_evaluable_outcome(dynamic_counts, span, dynamic)
        replay.add_paired_evaluable(paired, span, fixed, dynamic, frame_times)
    assert fixed_counts.hold_long_observed == 1
    assert fixed_counts.hold_long_observed_cutoffs == 1
    assert dynamic_counts.hold_long_observed == 0
    assert paired["hold_at_least_640ms"] == {
        "jointly_evaluable": 0,
        "not_jointly_observed_preceding_speech": 1,
        "not_observed_at_label_deadline": 0,
        "both_endpoint": 0,
        "fixed_only_endpoint": 0,
        "dynamic_only_endpoint": 0,
        "neither_endpoint": 0,
    }
    assert paired["eot_at_least_640ms"]["jointly_evaluable"] == 1
    assert paired["eot_at_least_640ms"]["both_endpoint"] == 1
    assert fixed_counts.eot_long_observed_detected == 1
    assert dynamic_counts.eot_long_observed_detected == 1


def test_jointly_observed_hold_without_frame_before_resume_is_not_evaluable() -> None:
    span = replay.LabeledSpan(1000, 1640, True)
    trace = replay.AcousticEndpointTrace((1600,), (992,))
    paired = replay._empty_paired_evaluable()
    replay.add_paired_evaluable(paired, span, trace, trace, (1640,))
    assert paired["hold_at_least_640ms"]["not_observed_at_label_deadline"] == 1
    assert paired["hold_at_least_640ms"]["jointly_evaluable"] == 0


def test_reused_silero_scorer_resets_recurrent_state_per_turn() -> None:
    np = pytest.importorskip("numpy")

    class FakeSession:
        def __init__(self) -> None:
            self.inputs: list[dict[str, object]] = []

        def run(self, _outputs: object, inputs: dict[str, object]) -> tuple[object, object]:
            self.inputs.append({key: value.copy() for key, value in inputs.items()})
            return np.array([[0.6]], dtype=np.float32), np.ones_like(inputs["state"])

    session = FakeSession()
    samples = np.full(1024, 0.1, dtype=np.float32)
    first = replay._silero_probabilities(samples, session)
    second = replay._silero_probabilities(samples, session)
    assert first == second
    assert [point.observed_at_ms for point in first] == [32, 64]
    assert len(session.inputs) == 4
    assert np.all(session.inputs[0]["state"] == 0)
    assert np.all(session.inputs[2]["state"] == 0)
    assert np.all(session.inputs[0]["input"][:, :64] == 0)
    assert np.all(session.inputs[2]["input"][:, :64] == 0)


def test_threshold_band_requires_consecutive_start_opportunity() -> None:
    configs = replay._configs()
    noise = [replay.AcousticPoint((i + 1) * 32, 0.1, -20.0) for i in range(4)]
    candidate = [replay.AcousticPoint((i + 5) * 32, 0.52, -40.0) for i in range(3)]
    counts, fixed, dynamic = replay.diagnose_start_opportunities(
        noise + candidate, configs["fixed"], configs["dynamic"]
    )
    assert counts.threshold_band_frames == 3
    assert counts.fixed_qualifying_windows_with_band == 1
    assert counts.fixed_only_start_transitions == 1
    assert fixed.active_frame_times_ms == (224,)
    assert dynamic.active_frame_times_ms == ()
    assert counts.summary()["dynamic_start_threshold_at_start_checks"]["max"] > 0.5
    assert counts.summary()["noise_floor_dbfs_at_start_checks"]["count"] == 7


def test_isolated_threshold_band_frame_is_not_a_start_window() -> None:
    configs = replay._configs()
    points = [
        *[replay.AcousticPoint((i + 1) * 32, 0.1, -20.0) for i in range(4)],
        replay.AcousticPoint(160, 0.52, -40.0),
        replay.AcousticPoint(192, 0.1, -20.0),
        replay.AcousticPoint(224, 0.52, -40.0),
    ]
    counts, _, _ = replay.diagnose_start_opportunities(points, configs["fixed"], configs["dynamic"])
    assert counts.threshold_band_frames == 2
    assert counts.fixed_qualifying_windows_with_band == 0
    assert counts.fixed_only_start_transitions == 0


@pytest.mark.parametrize(
    "spans",
    [
        [],
        [{"start": float("nan"), "end": 1.0}],
        [{"start": 1.0, "end": 0.9}],
        [{"start": 0.0, "end": 1.0}, {"start": 0.9, "end": 1.1}],
        [{"start": 0.0, "end": 3.0}],
    ],
)
def test_rejects_invalid_or_out_of_horizon_spans(spans: list[dict[str, float]]) -> None:
    with pytest.raises(ValueError):
        replay.eligible_spans(spans, audio_duration_ms=2000)


def test_fixed_and_dynamic_pairs_vary_only_noise_gain() -> None:
    configs = replay._configs()
    assert configs["fixed"].probability_gain_per_noise_db == 0.0
    assert configs["dynamic"].probability_gain_per_noise_db == 0.005
    assert not configs["fixed"].rearm_during_pause
    assert not configs["dynamic"].rearm_during_pause
    assert configs["fixed_rearm"].rearm_during_pause
    assert configs["dynamic_rearm"].rearm_during_pause
    assert configs["fixed_rearm"].probability_gain_per_noise_db == 0.0
    assert configs["dynamic_rearm"].probability_gain_per_noise_db == 0.005


def test_cli_refuses_to_overwrite_before_loading_optional_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "dataset.arrow"
    model_path = tmp_path / "silero.onnx"
    output_path = tmp_path / "aggregate.json"
    for path in (input_path, model_path, output_path):
        path.write_bytes(b"placeholder")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eot_bench_zh_replay.py",
            "--input",
            str(input_path),
            "--silero-model",
            str(model_path),
            "--output",
            str(output_path),
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        replay.main()
    assert output_path.read_bytes() == b"placeholder"
