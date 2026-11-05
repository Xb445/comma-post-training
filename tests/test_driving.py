import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from comma_post_training.driving.losses import action_distillation_loss
from comma_post_training.driving.manifest import REQUIRED_ARTIFACTS, validate_manifest
from comma_post_training.driving.metrics import evaluate_file, evaluate_record

EXAMPLES = Path(__file__).resolve().parents[1] / "data/examples/driving"


@pytest.fixture
def prediction():
    return json.loads((EXAMPLES / "predictions.jsonl").read_text().splitlines()[0])


def test_analytic_action_errors_jerk_and_trajectory_metrics(prediction):
    metrics = evaluate_record(prediction)
    assert metrics["lateral_accel_mae_mps2"] == pytest.approx(1.0)
    assert metrics["longitudinal_accel_mae_mps2"] == 0.0
    # Acceleration increases by 1 m/s^2 every 0.5 s: jerk=2, snap=0.
    assert metrics["lateral_jerk_rms_mps3"] == pytest.approx(2.0)
    assert metrics["trajectory_ade_m"] == pytest.approx(2 / 3)
    assert metrics["trajectory_fde_m"] == pytest.approx(1.0)


def test_nonuniform_action_times_and_independent_trajectory_grid(prediction):
    prediction["action_times_s"] = [0.0, 0.2, 1.0]
    prediction["predicted_actions"] = [[0, 0], [0.4, 0], [2, 0]]
    prediction["trajectory"]["times_s"] = [0, 0.1, 10]
    assert evaluate_record(prediction)["lateral_jerk_rms_mps3"] == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("action_times_s", [0, 0, 1]),
        ("action_times_s", [0, float("nan"), 1]),
        ("action_times_s", [-1, 0, 1]),
        ("predicted_actions", [[0, 0]]),
        ("predicted_actions", [[0, 0], [1, float("inf")], [2, 0]]),
        ("predicted_actions", [[0, 0, 0], [1, 0, 0], [2, 0, 0]]),
        ("coordinate_frame", "world"),
        ("action_order", ["a_long", "a_lat"]),
        ("action_units", "steering_degrees"),
    ],
)
def test_invalid_evaluation_contract_fails(prediction, key, value):
    prediction[key] = value
    with pytest.raises(ValueError):
        evaluate_record(prediction)


def test_synthetic_report_is_explicit_and_does_not_join_episode_boundaries():
    report = evaluate_file(EXAMPLES / "predictions.jsonl")
    assert report["source"] == "synthetic"
    assert report["mode"] == "open_loop"
    assert report["aggregation"] == "macro_per_record"
    assert report["records"] == report["routes"] == 2
    assert report["metrics"]["lateral_jerk_rms_mps3"] == pytest.approx(1.0)


@pytest.mark.parametrize("change", ["source", "sample_id", "trajectory_grid"])
def test_incomparable_or_duplicate_records_cannot_be_aggregated(tmp_path, prediction, change):
    second = copy.deepcopy(prediction)
    second["sample_id"] = "another"
    if change == "source":
        second["source"] = "recorded"
    elif change == "sample_id":
        second["sample_id"] = prediction["sample_id"]
    else:
        second["trajectory"]["times_s"] = [0, 0.3, 1]
    path = tmp_path / "predictions.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in (prediction, second)))
    with pytest.raises(ValueError):
        evaluate_file(path)


def test_manifest_detects_route_leakage_not_just_segment_duplicates(tmp_path):
    manifest = json.loads((EXAMPLES / "segments.json").read_text())
    manifest["segments"][1]["split"] = "test"
    path = tmp_path / "segments.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="route leakage"):
        validate_manifest(path, check_files=False)


def test_manifest_checks_recorded_revision_and_required_local_artifacts(tmp_path):
    manifest = json.loads((EXAMPLES / "segments.json").read_text())
    manifest.update(kind="recorded", dataset_revision="a" * 40)
    manifest["segments"] = [manifest["segments"][0]]
    manifest["segments"][0]["artifacts"] = {name: name for name in REQUIRED_ARTIFACTS}
    path = tmp_path / "segments.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="missing or empty"):
        validate_manifest(path)
    for name in REQUIRED_ARTIFACTS:
        (tmp_path / name).write_bytes(b"file-presence-test-only")
    assert validate_manifest(path)["asset_validation"] == "file_presence_only"
    (tmp_path / "frame_info.safetensors").write_bytes(b"")
    with pytest.raises(ValueError, match="frame_info.safetensors"):
        validate_manifest(path)
    manifest["dataset_revision"] = "main"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="immutable"):
        validate_manifest(path, check_files=False)


def test_metadata_only_validation_does_not_claim_assets_are_present():
    result = validate_manifest(EXAMPLES / "segments.json", check_files=False)
    assert result["asset_validation"] == "not_checked"
    assert result["segments"] == 4
    assert result["routes"] == 3
    assert result["split_segments"] == {"test": 1, "train": 2, "val": 1}


def test_bc_kd_weighting_masks_and_frozen_targets():
    prediction = torch.tensor([[0.0, 0.0], [1.0, 1.0]], requires_grad=True)
    expert = torch.tensor([[2.0, 2.0], [float("nan"), 0.0]], requires_grad=True)
    teacher = torch.tensor([[1.0, 1.0], [float("nan"), 0.0]], requires_grad=True)
    losses = action_distillation_loss(
        prediction,
        expert=expert,
        teacher=teacher,
        bc_weight=1.0,
        kd_weight=0.5,
        valid=torch.tensor([True, False]),
    )
    assert losses["total"].item() == pytest.approx(4.5)
    losses["total"].backward()
    assert prediction.grad.tolist() == [[-2.5, -2.5], [0.0, 0.0]]
    assert expert.grad is None and teacher.grad is None


@pytest.mark.parametrize("method", ["bc", "kd"])
def test_single_target_ablations_do_not_require_the_inactive_target(method):
    prediction = torch.zeros((2, 2), requires_grad=True)
    target = torch.ones_like(prediction)
    kwargs = (
        {"expert": target, "kd_weight": 0.0}
        if method == "bc"
        else {
            "teacher": target,
            "bc_weight": 0.0,
        }
    )
    assert action_distillation_loss(prediction, **kwargs)["total"].item() == pytest.approx(1.0)


def test_missing_supervision_and_empty_masks_fail():
    prediction = torch.zeros((2, 2))
    with pytest.raises(ValueError, match="target"):
        action_distillation_loss(prediction)
    with pytest.raises(ValueError, match="at least one"):
        action_distillation_loss(prediction, valid=torch.tensor([False, False]))
    with pytest.raises(ValueError, match="weights"):
        action_distillation_loss(prediction, bc_weight=0, kd_weight=0)


def test_cli_success_and_failure_exit_codes():
    # Exercise the installed public entry point rather than only importing functions.
    executable = Path(sys.executable).parent / "comma-driving"
    success = subprocess.run(
        [str(executable), "evaluate", str(EXAMPLES / "predictions.jsonl")],
        capture_output=True,
        text=True,
    )
    assert success.returncode == 0
    assert json.loads(success.stdout)["source"] == "synthetic"
    failure = subprocess.run(
        [str(executable), "validate-manifest", str(EXAMPLES / "segments.json")],
        capture_output=True,
        text=True,
    )
    assert failure.returncode == 2
    assert "missing or empty artifact" in failure.stderr
