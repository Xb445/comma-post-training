import json
import re

import numpy as np
import pytest
from PIL import Image

from comma_post_training.data_pipeline.demo import create_demo
from comma_post_training.data_pipeline.episodes import Catalogue, Episode
from comma_post_training.engine.checkpoints import file_sha256
from comma_post_training.reporting.inputs import (
    compare_runs,
    episode_values,
    load_predictions,
    load_run,
)
from comma_post_training.reporting.report import build_report

pytest.importorskip("matplotlib", reason="install the visualization extra for report tests")


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content))


def test_report_aligns_cameras_masks_and_predictions_without_changing_inputs(tmp_path):
    manifest = create_demo(tmp_path / "data", frames=12)
    catalogue = Catalogue(manifest)
    ref = catalogue.episodes[0]
    folder = ref.metadata.parent
    valid = np.ones(12, dtype=bool)
    valid[:3] = False
    np.save(folder / "teacher_valid.npy", valid)
    doc = json.loads(ref.metadata.read_text())
    doc["arrays"]["teacher_valid"] = "teacher_valid.npy"
    ref.metadata.write_text(json.dumps(doc))
    before = {p: file_sha256(p) for p in (manifest, ref.metadata, folder / "images.npy")}
    episode = Episode(ref)
    values = episode_values(episode)
    assert np.isnan(values["teacher"][:3]).all()
    prediction = tmp_path / "prediction"
    prediction.mkdir()
    row = {
        "episode_id": ref.id,
        "route_id": ref.route_id,
        "frame": 3,
        "timestamp_s": 0.6,
        "teacher_action": [1, 2],
        "adapted_action": [0.5, 1],
        "expert_action": [0, 0],
        "action_order": ["a_lat", "a_long"],
        "action_units": "m/s^2",
    }
    (prediction / "predictions.jsonl").write_text(json.dumps(row) + "\n")
    smoke = tmp_path / "smoke.json"
    write(
        smoke,
        {
            "teacher": {"kind": "synthetic_fixture"},
            "notes": "</script><script>alert('unsafe')</script>",
        },
    )
    report = tmp_path / "report"
    result = build_report(
        report,
        manifest=manifest,
        prediction_dirs=[prediction],
        smoke=smoke,
        max_episodes=1,
        max_frames=4,
        image_width=64,
        gif=True,
    )
    assert result["viewer_episodes"] == 1
    content = (report / "index.html").read_text()
    assert "<script>alert('unsafe')</script>" not in content
    payload = json.loads(re.search(r'id="report-data">(.*?)</script>', content, re.S)[1])
    viewer = payload["viewers"][0]
    assert viewer["camera_names"] == ["front_narrow", "front_wide"]
    assert viewer["indices"] == [0, 3, 7, 11]
    assert viewer["values"]["teacher"][0] == [None, None]
    assert viewer["values"]["adapted"][1] == [0.5, 1]
    assert viewer["images"][0][0].startswith("data:image/jpeg;base64,")
    assert viewer["images"][0][0] != viewer["images"][0][1]
    assert (report / "episode-000.gif").is_file()
    with Image.open(report / "episode-000.gif") as image:
        assert image.height == 16 + 24  # Preserve the source aspect; never crop to a model ratio.
    assert (report / "episode-000.csv").is_file()
    assert list((report / "figures").glob("*.png"))
    assert list((report / "figures").glob("*.svg"))
    assert all(file_sha256(path) == digest for path, digest in before.items())
    assert payload["summary"]["predictions"][0]["per_route"][ref.route_id]["adapted_mae_mps2"] == [
        0.5,
        1,
    ]
    with pytest.raises(ValueError, match="empty"):
        build_report(report, smoke=smoke)


def test_prediction_alignment_and_duplicate_frames_are_rejected(tmp_path):
    manifest = create_demo(tmp_path / "data", frames=12)
    references = {ref.id: Episode(ref) for ref in Catalogue(manifest).episodes}
    ref = next(iter(references.values())).reference
    folder = tmp_path / "prediction"
    folder.mkdir()
    row = {
        "episode_id": ref.id,
        "route_id": ref.route_id,
        "frame": 3,
        "timestamp_s": 0.65,
        "teacher_action": [0, 0],
        "adapted_action": [0, 0],
        "action_order": ["a_lat", "a_long"],
        "action_units": "m/s^2",
    }
    file = folder / "predictions.jsonl"
    file.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="align"):
        load_predictions(folder, references)
    row["timestamp_s"] = 0.6
    file.write_text(json.dumps(row) + "\n" + json.dumps(row))
    with pytest.raises(ValueError, match="duplicate"):
        load_predictions(folder, references)


def online_run(folder, *, scenario="route-a", after=-48.8):
    base = {
        "split": "val",
        "mean_return": -0.5,
        "mean_controlled_return": -0.4,
        "episodes": [
            {
                "seed": 42,
                "scenario_id": scenario,
                "controlled_return": -0.4,
                "controlled_steps": 50,
                "terminated": False,
                "truncated": True,
            }
        ],
    }
    final = {
        **base,
        "mean_controlled_return": after,
        "episodes": [
            {
                **base["episodes"][0],
                "controlled_return": after,
                "controlled_steps": 10,
                "terminated": True,
                "truncated": False,
            }
        ],
    }
    write(
        folder / "metrics.json",
        {
            "algorithm": "td3",
            "official_baseline": base,
            "initial_validation": base,
            "final_validation": final,
        },
    )
    return folder


def test_degradation_normalizes_steps_and_does_not_hide_termination_or_missing_logs(tmp_path):
    folder = online_run(tmp_path / "td3")
    run = load_run(folder)
    summary = run["summary"]
    assert summary["verdict"] == "degraded"
    assert summary["evaluations"]["final_validation"]["terminated"] == 1
    assert summary["evaluations"]["final_validation"][
        "controlled_return_per_step"
    ] == pytest.approx(-4.88)
    assert summary["events"] == 0
    assert not compare_runs([run, run])["same_recorded_setup"]
    output = tmp_path / "report"
    build_report(output, run_dirs=[folder])
    assert "No events.jsonl" in (output / "figures/training-00.svg").read_text()
    assert json.loads((output / "summary.json").read_text())["runs"][0]["verdict"] == "degraded"


def test_different_validation_scenarios_cannot_produce_a_paired_improvement_claim(tmp_path):
    folder = online_run(tmp_path / "run", after=-0.1)
    doc = json.loads((folder / "metrics.json").read_text())
    doc["final_validation"]["episodes"][0]["scenario_id"] = "different"
    write(folder / "metrics.json", doc)
    assert load_run(folder)["summary"]["verdict"] == "not_comparable"


def test_bad_logs_do_not_create_partial_reports(tmp_path):
    folder = online_run(tmp_path / "run")
    (folder / "events.jsonl").write_text('{"step": 1, "critic_loss": NaN}\n')
    output = tmp_path / "report"
    with pytest.raises(ValueError, match="nonfinite"):
        build_report(output, run_dirs=[folder])
    assert not output.exists()


@pytest.mark.parametrize("options", [{"max_episodes": 0}, {"max_frames": 1}, {"image_width": 0}])
def test_report_limits_are_explicit(tmp_path, options):
    with pytest.raises(ValueError):
        build_report(tmp_path / "report", **options)
