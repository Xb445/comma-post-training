import base64
import json
import pickle
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest
import torch

from comma_post_training.data_pipeline.demo import create_demo
from comma_post_training.data_pipeline.episodes import Catalogue, Episode
from comma_post_training.engine.checkpoints import file_sha256, load_checkpoint
from comma_post_training.engine.config import (
    DataConfig,
    LossConfig,
    ModelConfig,
    RunConfig,
    TrainingConfig,
)
from comma_post_training.engine.tasks import build_model
from comma_post_training.engine.trainer import evaluate, train
from comma_post_training.teacher.assets import ModelIdentity, verify_model
from comma_post_training.teacher.cache import cache_teacher, iter_teacher_episode, smoke_teacher
from comma_post_training.teacher.predict import predict_adapted
from comma_post_training.teacher.preprocess import RGBAdapter, pack_yuv420
from comma_post_training.teacher.runtime import TeacherSession, decode_output_slices

onnx = pytest.importorskip("onnx", reason="install the teacher extra for ONNX integration tests")
ort = pytest.importorskip(
    "onnxruntime", reason="install the teacher extra for ONNX integration tests"
)
cv2 = pytest.importorskip("cv2", reason="install the teacher extra for RGB preprocessing tests")


def make_fixture(
    path, *, action_width=2, fp16_controls=False, trailing_pad=False, official_layout=False
):
    """A tiny *synthetic* graph consuming every official-shaped input, not official weights."""
    from onnx import TensorProto, helper, numpy_helper

    if action_width not in (2, 4):
        raise ValueError("test fixture supports two/four raw action channels")
    if official_layout:
        action_width, fp16_controls, trailing_pad = 4, True, True
    features = 512 if official_layout else 4
    history = 24 if official_layout else 3
    desire_history = 25 if official_layout else 3
    height, width = (128, 256) if official_layout else (8, 16)
    prefix = 2062 if official_layout else 0
    control_type = TensorProto.FLOAT16 if fp16_controls else TensorProto.FLOAT
    inputs = [
        helper.make_tensor_value_info(name, kind, shape)
        for name, kind, shape in (
            ("img", TensorProto.UINT8, [1, 12, height, width]),
            ("big_img", TensorProto.UINT8, [1, 12, height, width]),
            ("features_buffer", control_type, [1, history, features]),
            ("desire_pulse", control_type, [1, desire_history, 8]),
            ("traffic_convention", control_type, [1, 2]),
            ("action_t", control_type, [1, 2]),
        )
    ]
    nodes = []
    for image in ("img", "big_img"):
        nodes.extend(
            [
                helper.make_node("Cast", [image], [image + "_float"], to=TensorProto.FLOAT),
                helper.make_node(
                    "ReduceMean", [image + "_float"], [image + "_mean"], axes=[1, 2, 3], keepdims=0
                ),
            ]
        )
    nodes.extend(
        [
            helper.make_node("Add", ["img_mean", "big_img_mean"], ["image_sum"]),
            helper.make_node("Mul", ["image_sum", "scale"], ["signal"]),
            helper.make_node("Unsqueeze", ["signal", "axis1"], ["signal_2d"]),
            helper.make_node("Add", ["signal_2d", "offsets"], ["features"]),
            helper.make_node(
                "ReduceMean", ["features_buffer"], ["history_mean"], axes=[1, 2], keepdims=0
            ),
            helper.make_node("Unsqueeze", ["history_mean", "axis1"], ["history_2d"]),
            helper.make_node(
                "ReduceMean", ["desire_pulse"], ["desire_mean"], axes=[1, 2], keepdims=0
            ),
            helper.make_node("Unsqueeze", ["desire_mean", "axis1"], ["desire_2d"]),
            helper.make_node("Mul", ["traffic_convention", "scale"], ["traffic"]),
            helper.make_node("Add", ["history_2d", "action_t"], ["action_base"]),
            helper.make_node("Add", ["action_base", "desire_2d"], ["action_desire"]),
            helper.make_node("Add", ["action_desire", "traffic"], ["action"]),
            helper.make_node("Concat", ["action", "features"], ["outputs"], axis=1),
        ]
    )
    # Keep arithmetic float32 on CPU while exercising real float16 graph inputs.
    if fp16_controls:
        controls = {"features_buffer", "desire_pulse", "traffic_convention", "action_t"}
        for node in nodes:
            for i, name in enumerate(node.input):
                if name in controls:
                    node.input[i] = name + "_float"
        nodes[:0] = [
            helper.make_node("Cast", [name], [name + "_float"], to=TensorProto.FLOAT)
            for name in sorted(controls)
        ]
    # The default graph retains the original tiny two-channel fixture contract.
    # The official-shaped graph recreates the reported slices, not official weights.
    concat = nodes[-1]
    output_parts = ["action"]
    if action_width == 4:
        output_parts.append("auxiliary")
    output_parts.append("features")
    if trailing_pad:
        output_parts.append("padding")
    if prefix:
        output_parts.insert(0, "prefix")
    concat.input[:] = output_parts
    constants = [
        numpy_helper.from_array(value, name)
        for name, value in (
            ("scale", np.array([0.001], dtype=np.float32)),
            ("axis1", np.array([1], dtype=np.int64)),
            ("offsets", np.linspace(0, 0.3, features, dtype=np.float32)[None]),
        )
    ]
    if action_width == 4:
        constants.append(
            numpy_helper.from_array(np.array([[123.0, -456.0]], dtype=np.float32), "auxiliary")
        )
    if trailing_pad:
        constants.append(
            numpy_helper.from_array(np.array([[777.0, 888.0]], dtype=np.float32), "padding")
        )
    if prefix:
        constants.append(numpy_helper.from_array(np.zeros((1, prefix), dtype=np.float32), "prefix"))
    output_width = prefix + action_width + features + (2 if trailing_pad else 0)
    graph = helper.make_graph(
        nodes,
        "synthetic_official_interface_fixture",
        inputs,
        [helper.make_tensor_value_info("outputs", TensorProto.FLOAT, [1, output_width])],
        constants,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    prop = model.metadata_props.add()
    prop.key = "output_slices"
    slices = {
        "action": slice(prefix, prefix + action_width),
        "hidden_state": slice(prefix + action_width, prefix + action_width + features),
    }
    if trailing_pad:
        slices["pad"] = slice(-2, None)
    if official_layout:
        slices.update(
            lane_lines=slice(0, 528),
            lane_lines_prob=slice(528, 536),
            road_edges=slice(536, 800),
            meta=slice(800, 855),
            desire_pred=slice(855, 887),
            pose=slice(887, 899),
            wide_from_device_euler=slice(899, 905),
            road_transform=slice(905, 917),
            plan=slice(917, 1907),
            lead=slice(1907, 2051),
            lead_prob=slice(2051, 2054),
            desire_state=slice(2054, 2062),
        )
    prop.value = base64.b64encode(pickle.dumps(slices)).decode()
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return ModelIdentity(
        "synthetic_fixture",
        "local-test-graph",
        file_sha256(path),
        path.stat().st_size,
        "local-fixture",
    )


@pytest.fixture(scope="module")
def teacher_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("teacher-fixture")
    path = root / "synthetic.onnx"
    identity = make_fixture(path)
    return path, identity


def teacher(teacher_fixture):
    path, identity = teacher_fixture
    return TeacherSession(path, identity=identity)


def attach_profiles(manifest):
    for ref in Catalogue(manifest).episodes:
        doc = json.loads(ref.metadata.read_text())
        doc["teacher_rgb"] = {
            "format": "openpilot-supercombo-rgb-v1",
            "mode": "prewarped_model_rgb",
            "calibration_id": "synthetic-camera-fixture-not-real-calibration",
            "camera_mapping": {"img": "front_narrow", "big_img": "front_wide"},
            "context_hz": 5,
            "desire_mode": "none",
            "traffic_convention": [1, 0],
            "action_t_s": [0.2, 0.2],
        }
        ref.metadata.write_text(json.dumps(doc))


def test_weights_are_verified_and_fixture_never_claims_official(teacher_fixture, tmp_path):
    path, identity = teacher_fixture
    assert verify_model(path, identity)["kind"] == "synthetic_fixture"
    with pytest.raises(ValueError, match="size differs"):
        verify_model(path)
    changed = tmp_path / "changed.onnx"
    contents = bytearray(path.read_bytes())
    contents[-1] ^= 1
    changed.write_bytes(contents)
    with pytest.raises(ValueError, match="SHA256"):
        verify_model(changed, identity)


def test_cli_cannot_treat_a_synthetic_graph_as_official_weights(teacher_fixture):
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "comma_post_training.teacher.cli",
            "verify",
            "--model",
            str(teacher_fixture[0]),
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 2
    assert "size differs" in process.stderr
    assert "Traceback" not in process.stderr


def test_output_metadata_rejects_pickle_code_execution():
    payload = base64.b64encode(b"cos\nsystem\n(S'echo unsafe'\ntR.").decode()
    with pytest.raises(ValueError, match="only primitives"):
        decode_output_slices(payload, 100)
    outside = base64.b64encode(pickle.dumps({"action": slice(0, 101)})).decode()
    with pytest.raises(ValueError, match="outside"):
        decode_output_slices(outside, 100)


def test_yuv_pack_matches_pinned_source_plane_order():
    rgb = np.random.default_rng(9).integers(0, 256, (16, 32, 3), dtype=np.uint8)
    yuv = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV_I420)
    upstream_y = yuv[:16].reshape(8, 2, 16, 2).transpose(3, 1, 0, 2).reshape(4, 8, 16)
    expected = np.concatenate((upstream_y, yuv[16:].reshape(2, 8, 16)))
    np.testing.assert_array_equal(pack_yuv420(rgb), expected)
    with pytest.raises(ValueError, match="even"):
        pack_yuv420(np.zeros((15, 32, 3), dtype=np.uint8))


def test_streaming_state_matches_original_onnx_and_resets(teacher_fixture):
    session = teacher(teacher_fixture)
    image = pack_yuv420(np.full((16, 32, 3), 80, dtype=np.uint8))
    # Independent ORT execution of the unmodified graph at the first frame.
    feed = {
        name: np.zeros(shape, dtype=session.dtypes[name]) for name, shape in session.shapes.items()
    }
    for key in ("img", "big_img"):
        feed[key][:, 6:] = image
    feed["traffic_convention"][0] = [1, 0]
    feed["action_t"][0] = [0.2, 0.2]
    raw = session.session.run([session.output_name], feed)[0]
    controls = {"desire": np.zeros(8), "traffic": [1, 0], "action_t": [0.2, 0.2]}
    first = session.step({"img": image, "big_img": image}, **controls)
    np.testing.assert_allclose(first["action"], raw[0, :2])
    np.testing.assert_allclose(first["features"], raw[0, 2:])
    second = session.step({"img": image, "big_img": image}, **controls)
    assert not np.array_equal(first["action"], second["action"])
    session.reset()
    reset = session.step({"img": image, "big_img": image}, **controls)
    np.testing.assert_array_equal(first["action"], reset["action"])
    flags = [reset["valid"]]
    for _ in range(session.warmup_frames):
        flags.append(session.step({"img": image, "big_img": image}, **controls)["valid"])
    assert flags == [False] * session.warmup_frames + [True]


def test_desire_is_a_rising_edge_pulse_not_repeated_state(teacher_fixture):
    session = teacher(teacher_fixture)
    image = pack_yuv420(np.zeros((16, 32, 3), dtype=np.uint8))
    desire = np.zeros(8)
    desire[3] = 1
    inputs = {"img": image, "big_img": image}
    for _ in range(2):
        session.step(inputs, desire=desire, traffic=[1, 0], action_t=[0.2, 0.2])
    np.testing.assert_array_equal(session.queues["desire_pulse"][0, -2:, 3], [1, 0])
    with pytest.raises(ValueError, match="traffic"):
        session.step(inputs, desire=desire, traffic=[0, 0], action_t=[0.2, 0.2])


def test_smoke_is_explicitly_synthetic_input(teacher_fixture):
    report = smoke_teacher(teacher(teacher_fixture), frames=2)
    assert report["teacher"]["kind"] == "synthetic_fixture"
    assert report["reset_reproducible"]
    assert not report["real_driving_validation"]
    assert not any(report["warmed"])


def test_cache_requires_calibration_and_rejects_silent_image_resize(teacher_fixture, tmp_path):
    manifest = create_demo(tmp_path / "data", frames=12)
    episode = Episode(Catalogue(manifest).episodes[0])
    with pytest.raises(ValueError, match="teacher_rgb"):
        RGBAdapter(episode, (16, 32))
    attach_profiles(manifest)
    episode = Episode(Catalogue(manifest).episodes[0])
    with pytest.raises(ValueError, match="resizing is not"):
        RGBAdapter(episode, (32, 64)).images(0)
    doc = json.loads(episode.reference.metadata.read_text())
    doc["teacher_rgb"].update(
        mode="source_to_model_homography", source_to_model={"img": None, "big_img": None}
    )
    episode.reference.metadata.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="homographies"):
        RGBAdapter(Episode(episode.reference), (16, 32))


def test_cache_resets_at_invalid_frames_and_cannot_accept_20hz(teacher_fixture, tmp_path):
    manifest = create_demo(tmp_path / "data", frames=16)
    attach_profiles(manifest)
    ref = Catalogue(manifest).episodes[0]
    mask = np.ones(16, dtype=bool)
    mask[7] = False
    np.save(ref.metadata.parent / "valid.npy", mask)
    outputs = list(iter_teacher_episode(Episode(ref), teacher(teacher_fixture)))
    assert outputs[7][1] is None
    assert outputs[8][1]["frame"] == 0
    assert not outputs[8][1]["valid"]
    np.save(ref.metadata.parent / "timestamps.npy", np.arange(16) * 0.05)
    with pytest.raises(ValueError, match="faster than 5 Hz"):
        list(iter_teacher_episode(Episode(ref), teacher(teacher_fixture)))


def test_official_feature_cache_residual_training_and_live_prediction(teacher_fixture, tmp_path):
    manifest = create_demo(tmp_path / "data", frames=20)
    attach_profiles(manifest)
    before = {ref.metadata: ref.metadata.read_bytes() for ref in Catalogue(manifest).episodes}
    session = teacher(teacher_fixture)
    model_before = file_sha256(teacher_fixture[0])
    report = cache_teacher(manifest, session, tmp_path / "cache")
    cached = Catalogue(tmp_path / "cache/manifest.json")
    assert report["teacher_identity"]["kind"] == "synthetic_fixture"
    assert all(count == 16 for count in report["valid_frames"].values())
    assert all(path.read_bytes() == content for path, content in before.items())
    for ref in cached.episodes:
        episode = Episode(ref)
        targets = np.array(episode.array("teacher_actions")) + np.array(
            [0.15, -0.1], dtype=np.float32
        )
        np.save(ref.metadata.parent / "expert-own.npy", targets)
        doc = json.loads(ref.metadata.read_text())
        doc["arrays"]["expert_actions"] = "expert-own.npy"
        ref.metadata.write_text(json.dumps(doc))
    config = RunConfig(
        "official_residual",
        DataConfig(tmp_path / "cache/manifest.json", history=1, image_size=(16, 32)),
        TrainingConfig(
            tmp_path / "adapt", max_steps=24, batch_size=8, val_every=8, learning_rate=0.005
        ),
        ModelConfig(hidden_size=16),
        LossConfig(bc_weight=1, kd_weight=0.1),
    )
    training = train(config)
    assert training["teacher_identity"]["kind"] == "synthetic_fixture"
    assert training["base_frozen"]
    assert training["final_validation"]["loss"] < training["initial_validation"]["loss"]
    partial = replace(
        config, training=replace(config.training, output_dir=tmp_path / "resumed", max_steps=8)
    )
    train(partial)
    continued = replace(config, training=replace(config.training, output_dir=tmp_path / "resumed"))
    train(continued, resume=tmp_path / "resumed/last.pt")
    resumed = load_checkpoint(tmp_path / "resumed/last.pt")
    assert evaluate(tmp_path / "adapt/last.pt")["metrics"]["expert_mse"] >= 0
    saved = load_checkpoint(tmp_path / "adapt/last.pt")
    for name, value in saved["model_state"].items():
        torch.testing.assert_close(value, resumed["model_state"][name], rtol=0, atol=0)
    assert saved["model_spec"]["teacher_identity"] == session.cache_identity()
    initial = load_checkpoint(tmp_path / "adapt/initial.pt")
    model = build_model(initial["model_spec"])
    model.load_state_dict(initial["model_state"])
    batch = {
        "teacher": torch.randn(4, 2),
        "teacher_features": torch.randn(4, 4),
        "state": torch.randn(4, 1, 2),
        "past_actions": torch.randn(4, 1, 2),
    }
    torch.testing.assert_close(model(batch), batch["teacher"], rtol=0, atol=0)
    model.load_state_dict(saved["model_state"])
    assert ((model(batch) - batch["teacher"]).abs() <= 0.5 + 1e-6).all()
    prediction = predict_adapted(
        tmp_path / "cache/manifest.json",
        tmp_path / "adapt/last.pt",
        session,
        tmp_path / "prediction",
    )
    assert prediction["frames"] == 32
    assert prediction["adapted_mae_mps2"][0] < prediction["teacher_mae_mps2"][0]
    assert file_sha256(teacher_fixture[0]) == model_before
    assert set(saved["model_state"]) == set(model.state_dict())
    assert not any(name.startswith("encoder") for name in saved["model_state"])
    # An asset/adapter mismatch must stop inference before producing output.
    session.identity = replace(session.identity, sha256="0" * 64)
    with pytest.raises(ValueError, match="exact frozen Teacher"):
        predict_adapted(manifest, tmp_path / "adapt/last.pt", session, tmp_path / "wrong")
    assert not (tmp_path / "wrong").exists()
    with pytest.raises(ValueError, match="expert BC"):
        replace(config, loss=LossConfig(bc_weight=0, kd_weight=1))


def test_failed_teacher_cache_does_not_publish_partial_dataset(teacher_fixture, tmp_path):
    manifest = create_demo(tmp_path / "data", frames=12)
    attach_profiles(manifest)
    ref = Catalogue(manifest).episodes[0]
    np.save(ref.metadata.parent / "timestamps.npy", np.arange(12) * 0.05)
    with pytest.raises(ValueError, match="resample"):
        cache_teacher(manifest, teacher(teacher_fixture), tmp_path / "failed")
    assert not (tmp_path / "failed").exists()


def test_real_report_layout_and_fp16_history_keep_auxiliary_out_of_actions(tmp_path):
    path = tmp_path / "official-layout-synthetic.onnx"
    identity = make_fixture(path, official_layout=True)
    session = TeacherSession(path, identity=identity)
    description = session.describe()
    assert description["kind"] == "synthetic_fixture"
    assert description["adapter_version"].endswith("-v2")
    assert description["input_shapes"]["features_buffer"] == (1, 24, 512)
    assert description["input_dtypes"]["features_buffer"] == "float16"
    assert description["output_width"] == 2580
    assert description["output_slices"]["action"] == [2062, 2066]
    assert description["output_slices"]["hidden_state"] == [2066, 2578]
    assert description["output_slices"]["pad"] == [2578, 2580]
    assert description["action_layout"]["raw_width"] == 4
    assert description["action_layout"]["mean_indices"] == [0, 1]
    image = pack_yuv420(np.full((256, 512, 3), 90, dtype=np.uint8))
    controls = {"desire": np.zeros(8), "traffic": [1, 0], "action_t": [0.2, 0.3]}
    feed = {
        name: np.zeros(shape, dtype=session.dtypes[name]) for name, shape in session.shapes.items()
    }
    for name in ("img", "big_img"):
        feed[name][:, 6:] = image
    feed["traffic_convention"][0] = controls["traffic"]
    feed["action_t"][0] = controls["action_t"]
    reference = session.session.run([session.output_name], feed)[0][0]
    prediction = session.step({"img": image, "big_img": image}, **controls)
    np.testing.assert_array_equal(prediction["raw_action"], reference[2062:2066])
    np.testing.assert_array_equal(prediction["action"], reference[2062:2064])
    np.testing.assert_array_equal(prediction["action_auxiliary"], [123.0, -456.0])
    np.testing.assert_array_equal(prediction["features"], reference[2066:2578])
    np.testing.assert_array_equal(
        session.queues["features_buffer"][0, -1], prediction["features"].astype(np.float16)
    )
    assert prediction["plan"].shape == (33, 15)
    assert session.warmup_frames == 25


def test_padding_and_negative_bounds_follow_valid_python_slices():
    metadata = {"action": slice(0, 4), "hidden_state": slice(4, -2), "pad": slice(-2, None)}
    encoded = base64.b64encode(pickle.dumps(metadata)).decode()
    slices = decode_output_slices(encoded, 10)
    for name, value in metadata.items():
        np.testing.assert_array_equal(np.arange(10)[slices[name]], np.arange(10)[value])
    assert slices["pad"] == slice(8, 10, 1) or slices["pad"] == slice(8, 10)


@pytest.mark.parametrize(
    "value",
    [slice(0, 11), slice(-11, None), slice(5, 4), slice(0, 4, 0), slice(0, 4, -1), slice(0.5, 4)],
)
def test_invalid_output_slices_are_rejected(value):
    encoded = base64.b64encode(pickle.dumps({"action": value})).decode()
    with pytest.raises(ValueError):
        decode_output_slices(encoded, 10)


def test_smoke_replays_full_sequence_and_reports_warmed_latency(teacher_fixture):
    session = teacher(teacher_fixture)
    frames = session.warmup_frames + 3
    report = smoke_teacher(session, frames=frames)
    assert report["warmed_frames"] == 3
    assert report["reset_sequence_max_abs"] == {"action": 0.0, "features": 0.0}
    assert report["latency_summary_ms"]["all"]["count"] == frames
    warmed = report["latency_summary_ms"]["warmed"]
    assert warmed["count"] == 3
    assert 0 <= warmed["p50"] <= warmed["p95"] <= warmed["p99"]


def test_smoke_detects_divergence_after_reproducible_first_frame(teacher_fixture, monkeypatch):
    session = teacher(teacher_fixture)
    original_reset, original_step = session.reset, session.step
    resets = 0

    def reset():
        nonlocal resets
        resets += 1
        original_reset()

    def step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        if resets >= 2 and result["frame"] > 0:
            result["action"] = result["action"] + 1.0
        return result

    monkeypatch.setattr(session, "reset", reset)
    monkeypatch.setattr(session, "step", step)
    with pytest.raises(RuntimeError, match="reset"):
        smoke_teacher(session, frames=3)


def test_ort_profile_reports_actual_node_provider(teacher_fixture, tmp_path):
    path, identity = teacher_fixture
    session = TeacherSession(path, identity=identity, profile_prefix=tmp_path / "ort-profile")
    report = smoke_teacher(session, frames=2)
    profile = report["profiling"]
    assert profile["providers"]["CPUExecutionProvider"]["node_events"] > 0
    assert profile["providers"]["CPUExecutionProvider"]["total_duration_us"] >= 0
    with open(profile["path"], encoding="utf-8") as stream:
        events = json.load(stream)
    assert any(event.get("args", {}).get("provider") == "CPUExecutionProvider" for event in events)
    assert session.finish_profiling() == profile


def test_cuda_request_rejects_runtime_silent_cpu_fallback(teacher_fixture, monkeypatch):
    path, identity = teacher_fixture
    original = ort.InferenceSession
    monkeypatch.setattr(
        ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    def fallback(model, *, sess_options, providers):
        assert providers == ["CUDAExecutionProvider"]
        return original(model, sess_options=sess_options, providers=["CPUExecutionProvider"])

    monkeypatch.setattr(ort, "InferenceSession", fallback)
    with pytest.raises(ValueError, match="silent CPU provider fallback"):
        TeacherSession(path, identity=identity, provider="cuda")
