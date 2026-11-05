import json

import numpy as np
import pytest
from PIL import Image

from comma_post_training.data_pipeline.demo import create_demo
from comma_post_training.data_pipeline.episodes import Catalogue, Episode
from comma_post_training.data_pipeline.geometry import (
    flu_to_frd_poses,
    motion_vector,
    relative_pose,
    validate_poses,
)
from comma_post_training.data_pipeline.prepare import build_manifest, import_npz
from comma_post_training.data_pipeline.windows import EpisodeWindows, StepBatchSampler


@pytest.fixture
def episodes(tmp_path):
    return create_demo(tmp_path / "data", frames=16)


def rewrite(path, change):
    document = json.loads(path.read_text())
    change(document)
    path.write_text(json.dumps(document))


def test_route_leakage_rejected_before_training(episodes):
    def leak(document):
        document["episodes"][2]["route_id"] = document["episodes"][0]["route_id"]

    rewrite(episodes, leak)
    with pytest.raises(ValueError, match="route leakage"):
        Catalogue(episodes)


@pytest.mark.parametrize(
    "field,value",
    [
        ("timestamp_units", "ms"),
        ("coordinate_frame", "vehicle_flu"),
        ("action_units", "steering_angle"),
    ],
)
def test_units_and_coordinates_must_be_explicit(episodes, field, value):
    ref = Catalogue(episodes).episodes[0]
    rewrite(ref.metadata, lambda doc: doc.update({field: value}))
    with pytest.raises(ValueError):
        Episode(ref).validate()


def test_invalid_label_nan_is_masked_and_current_action_is_not_an_input(episodes):
    catalogue = Catalogue(episodes)
    ref = catalogue.episodes[0]
    folder = ref.metadata.parent
    labels = np.load(folder / "expert_actions.npy")
    labels[4] = np.nan
    np.save(folder / "expert_actions.npy", labels)
    mask = np.ones(len(labels), dtype=bool)
    mask[4] = False
    np.save(folder / "expert_valid.npy", mask)
    rewrite(ref.metadata, lambda doc: doc["arrays"].update(expert_valid="expert_valid.npy"))
    dataset = EpisodeWindows(Catalogue(episodes), split="train", task="policy", history=3)
    assert all(
        not (dataset[i]["episode_id"] == ref.id and dataset[i]["frame"] == 4)
        for i in range(len(dataset))
    )
    sample = dataset[0]
    assert sample["frame"] == 2
    executed = np.load(folder / "executed_actions.npy")
    np.testing.assert_array_equal(sample["past_actions"][0], [0, 0])
    np.testing.assert_array_equal(sample["past_actions"][1:], executed[:2])
    np.testing.assert_array_equal(sample["expert"], executed[2])


def test_windows_do_not_cross_invalid_frames_or_time_gaps(episodes):
    ref = Catalogue(episodes).episodes[0]
    folder = ref.metadata.parent
    valid = np.ones(16, dtype=bool)
    valid[5] = False
    np.save(folder / "valid.npy", valid)
    times = np.load(folder / "timestamps.npy")
    times[11:] += 2
    np.save(folder / "timestamps.npy", times)
    dataset = EpisodeWindows(Catalogue(episodes), split="train", task="policy", history=3)
    samples = [dataset[i] for i in range(len(dataset)) if dataset[i]["episode_id"] == ref.id]
    assert [item["frame"] for item in samples] == [2, 3, 4, 8, 9, 10, 13, 14, 15]
    # First history frame after each discontinuity has no cross-gap action.
    for sample in samples:
        if sample["frame"] in (8, 13):
            np.testing.assert_array_equal(sample["past_actions"][0], [0, 0])


def test_stat_fingerprint_detects_changed_arrays(episodes):
    old = Catalogue(episodes).fingerprint
    ref = Catalogue(episodes).episodes[0]
    actions = np.load(ref.metadata.parent / "executed_actions.npy")
    actions[0, 0] += 1
    np.save(ref.metadata.parent / "executed_actions.npy", actions)
    assert Catalogue(episodes).fingerprint != old


def test_domain_sampling_and_step_resume_are_reproducible(episodes):
    document = json.loads(episodes.read_text())
    for i, row in enumerate(document["episodes"]):
        row["domain"] = ("real", "sim")[i % 2]
        rewrite(
            episodes.parent / row["metadata"],
            lambda doc, row=row: doc["episode"].update(domain=row["domain"]),
        )
    episodes.write_text(json.dumps(document))
    dataset = EpisodeWindows(Catalogue(episodes), split="train", task="policy", history=3)
    settings = dict(seed=8, steps=8, batch_size=12, domain_mix={"sim": 1.0})
    full = list(StepBatchSampler(dataset, start=0, **settings))
    assert full[3:] == list(StepBatchSampler(dataset, start=3, **settings))
    assert all(dataset[index]["domain"] == "sim" for batch in full for index in batch)
    mixed = dict(settings, domain_mix={"real": 0.7, "sim": 0.3})
    reverse = dict(settings, domain_mix={"sim": 0.3, "real": 0.7})
    assert list(StepBatchSampler(dataset, start=0, **mixed)) == list(
        StepBatchSampler(dataset, start=0, **reverse)
    )
    with pytest.raises(ValueError, match="available training domains"):
        StepBatchSampler(dataset, start=0, **{**settings, "domain_mix": {"synthetic": 1}})


def test_npz_import_lazy_images_and_failed_import_are_transactional(episodes, tmp_path):
    ref = Catalogue(episodes).episodes[0]
    original = Episode(ref)
    arrays = {name: np.array(original.array(name)) for name in original.paths if name != "images"}
    archive = tmp_path / "aligned.npz"
    np.savez(archive, **arrays)
    images = []
    for frame in range(16):
        row = []
        for view in range(2):
            path = tmp_path / f"frame-{frame}-{view}.png"
            Image.fromarray(np.array(original.array("images")[frame, view])).save(path)
            row.append(path.name)
        images.append(row)
    (tmp_path / "frames.json").write_text(json.dumps(images))
    metadata = dict(original.metadata)
    metadata.pop("arrays")
    metadata["image_paths"] = "frames.json"
    metadata["episode"] = dict(metadata["episode"], id="own-001", route_id="own-route-001")
    meta_path = tmp_path / "metadata.json"
    meta_path.write_text(json.dumps(metadata))
    root = tmp_path / "imported"
    result = import_npz(archive, meta_path, root / "episodes" / "own-001")
    assert result["frames"] == 16
    report = build_manifest(root, revision="own-dataset-v1")
    dataset = EpisodeWindows(
        Catalogue(root / "manifest.json"), split="train", task="policy", image_size=(16, 32)
    )
    np.testing.assert_allclose(dataset[0]["images"][0], original.images([0], (16, 32))[0])
    assert report["kind"] == "recorded"
    arrays["timestamps"][1] = arrays["timestamps"][0]
    np.savez(archive, **arrays)
    failed = tmp_path / "failed"
    with pytest.raises(ValueError, match="strictly increasing"):
        import_npz(archive, meta_path, failed)
    assert not failed.exists()
    with pytest.raises(ValueError, match="exists"):
        build_manifest(root, revision="overwrite")


def test_relative_se3_and_flu_to_frd_conversion():
    poses = np.tile(np.eye(4), (2, 1, 1))
    poses[0, :2, 3] = [10, 3]
    poses[1, :2, 3] = [12, 4]
    delta = relative_pose(poses[0], poses[1])
    np.testing.assert_allclose(motion_vector(poses[0], poses[1]), [2, 1, 0, 0, 0, 0])
    converted = flu_to_frd_poses(poses)
    np.testing.assert_allclose(relative_pose(converted[0], converted[1])[:3, 3], [2, -1, 0])
    np.testing.assert_allclose(delta[:3, 3], [2, 1, 0])
    poses[1, 0, 0] = 2
    with pytest.raises(ValueError, match="rotation"):
        validate_poses(poses)
