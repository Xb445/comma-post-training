"""Run the original ONNX graph, with explicit 5 Hz image/feature/desire history."""

import base64
import io
import pickle
import time
from pathlib import Path

import numpy as np

from comma_post_training.teacher.assets import OFFICIAL, ModelIdentity, verify_model

ADAPTER_VERSION = "supercombo-084747-stream-5hz-v1"
IMAGE_NAMES = ("img", "big_img")
INPUT_NAMES = {*IMAGE_NAMES, "features_buffer", "desire_pulse", "traffic_convention", "action_t"}
DTYPES = {"tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(uint8)": np.uint8}


class _SliceUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "builtins" and name == "slice":
            return slice
        raise ValueError("output_slices metadata may contain only primitives and builtins.slice")


def decode_output_slices(encoded: str, width: int) -> dict[str, slice]:
    try:
        payload = base64.b64decode(encoded, validate=True)
        if len(payload) > 1024 * 1024:
            raise ValueError("oversized output_slices metadata")
        result = _SliceUnpickler(io.BytesIO(payload)).load()
    except (pickle.UnpicklingError, EOFError, TypeError) as error:
        raise ValueError("invalid output_slices metadata") from error
    if not isinstance(result, dict) or not result:
        raise ValueError("output_slices must be a nonempty dictionary")
    for name, value in result.items():
        if not isinstance(name, str) or not isinstance(value, slice):
            raise ValueError("output_slices must map names to slices")
        if value.step not in (None, 1) or any(
            type(v) is not int for v in (value.start, value.stop)
        ):
            raise ValueError("output slices require explicit integer bounds and unit stride")
        if not 0 <= value.start <= value.stop <= width or (
            value.start == value.stop and name != "pad"
        ):
            raise ValueError("output slice is outside model outputs")
    return result


class TeacherSession:
    def __init__(
        self, model: Path, *, provider="cpu", threads=2, identity: ModelIdentity = OFFICIAL
    ):
        self.receipt = verify_model(model, identity)
        self.identity = identity
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise ValueError(
                "install Teacher dependencies: uv sync --locked --extra teacher"
            ) from error
        if provider not in ("cpu", "cuda") or type(threads) is not int or threads < 1:
            raise ValueError("provider must be cpu/cuda and threads must be positive")
        name = "CPUExecutionProvider" if provider == "cpu" else "CUDAExecutionProvider"
        if name not in ort.get_available_providers():
            raise ValueError(f"requested {name} is unavailable; no provider fallback is assumed")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        try:
            self.session = ort.InferenceSession(str(model), sess_options=options, providers=[name])
        except Exception as error:
            raise ValueError(
                f"ONNX Runtime could not load the verified Teacher: {error}"
            ) from error
        self.inputs = {value.name: value for value in self.session.get_inputs()}
        if set(self.inputs) != INPUT_NAMES:
            raise ValueError(f"unsupported Supercombo input names: {sorted(self.inputs)}")
        self.shapes, self.dtypes = {}, {}
        for key, info in self.inputs.items():
            if (
                not info.shape
                or info.shape[0] != 1
                or any(type(v) is not int or v < 1 for v in info.shape)
            ):
                raise ValueError("Teacher adapter requires the original static batch-1 graph")
            if info.type not in DTYPES:
                raise ValueError(f"unsupported Teacher input dtype: {info.type}")
            self.shapes[key], self.dtypes[key] = tuple(info.shape), DTYPES[info.type]
        for key in IMAGE_NAMES:
            shape = self.shapes[key]
            if len(shape) != 4 or shape[1] != 12:
                raise ValueError("Teacher img/big_img must pack two six-channel YUV420 frames")
        if self.shapes["img"] != self.shapes["big_img"]:
            raise ValueError("this pinned adapter requires equal camera tensor dimensions")
        if len(self.shapes["features_buffer"]) != 3 or self.shapes["desire_pulse"][2:] != (8,):
            raise ValueError("unsupported feature/desire queue layout")
        if self.shapes["traffic_convention"] != (1, 2) or self.shapes["action_t"] != (1, 2):
            raise ValueError("unsupported traffic/action_t layout")
        outputs = self.session.get_outputs()
        if len(outputs) != 1 or len(outputs[0].shape) != 2 or outputs[0].shape[0] != 1:
            raise ValueError("Teacher must return one packed [1,D] output")
        self.output_name = outputs[0].name
        if type(outputs[0].shape[-1]) is not int or outputs[0].shape[-1] < 1:
            raise ValueError("Teacher output width must be static and positive")
        encoded = self.session.get_modelmeta().custom_metadata_map.get("output_slices")
        if not isinstance(encoded, str):
            raise ValueError("Teacher is missing its output_slices metadata")
        self.slices = decode_output_slices(encoded, outputs[0].shape[-1])
        if not {"action", "hidden_state"}.issubset(self.slices):
            raise ValueError("pinned Teacher must expose action and hidden_state")
        self.feature_dim = self.shapes["features_buffer"][-1]
        if self.slices["hidden_state"].stop - self.slices["hidden_state"].start != self.feature_dim:
            raise ValueError("hidden_state does not match the recurrent feature queue")
        if self.slices["action"].stop - self.slices["action"].start != 2:
            raise ValueError("action output must be [a_lat,a_long], not an assumed plan conversion")
        self.image_size = (self.shapes["img"][2] * 2, self.shapes["img"][3] * 2)
        self.warmup_frames = max(
            self.shapes["features_buffer"][1] + 1, self.shapes["desire_pulse"][1] - 1
        )
        self.reset()

    def reset(self):
        self.queues = {
            name: np.zeros(shape, dtype=self.dtypes[name]) for name, shape in self.shapes.items()
        }
        self.previous_desire = np.zeros(8, dtype=np.float32)
        self.steps = 0

    def describe(self):
        return {
            **self.receipt,
            "adapter_version": ADAPTER_VERSION,
            "providers": self.session.get_providers(),
            "input_shapes": self.shapes,
            "input_dtypes": {k: np.dtype(v).name for k, v in self.dtypes.items()},
            "feature_dim": self.feature_dim,
            "context_hz": 5,
            "warmup_frames": self.warmup_frames,
            "action_semantics": "unsmoothed_network_a_lat_a_long_mps2",
            "output_slices": {key: [value.start, value.stop] for key, value in self.slices.items()},
        }

    def cache_identity(self):
        return {
            "kind": self.identity.kind,
            "commit": self.identity.commit,
            "sha256": self.identity.sha256,
            "adapter_version": ADAPTER_VERSION,
            "feature_dim": self.feature_dim,
            "action_semantics": "unsmoothed_network_a_lat_a_long_mps2",
        }

    def step(self, images: dict, *, desire, traffic, action_t):
        prepared = {}
        for name in IMAGE_NAMES:
            current = np.asarray(images.get(name))
            expected = (6, *self.shapes[name][2:])
            if current.shape != expected or current.dtype != np.uint8:
                raise ValueError(f"Teacher {name} must be a packed YUV420 uint8 {expected}")
            prepared[name] = current
        desire, traffic, action_t = (
            np.asarray(v, dtype=np.float32) for v in (desire, traffic, action_t)
        )
        if desire.shape != (8,) or traffic.shape != (2,) or action_t.shape != (2,):
            raise ValueError("desire/traffic/action_t shapes must be [8]/[2]/[2]")
        if not all(np.isfinite(v).all() for v in (desire, traffic, action_t)):
            raise ValueError("Teacher controls must be finite")
        if not np.all((desire == 0) | (desire == 1)) or desire.sum() > 1:
            raise ValueError("desire must be a one-hot state or all-zero no-intent state")
        if (
            not np.all((traffic == 0) | (traffic == 1))
            or traffic.sum() != 1
            or (action_t < 0).any()
        ):
            raise ValueError("traffic must be one-hot and action delays nonnegative seconds")
        for name, current in prepared.items():
            self.queues[name][:, :6] = self.queues[name][:, 6:].copy()
            self.queues[name][:, 6:] = current
        desire = desire.copy()
        desire[0] = 0
        pulse = np.where(desire - self.previous_desire > 0.99, desire, 0)
        self.previous_desire = desire
        queue = self.queues["desire_pulse"]
        queue[:, :-1] = queue[:, 1:].copy()
        queue[:, -1] = pulse
        self.queues["traffic_convention"][0] = traffic
        self.queues["action_t"][0] = action_t
        start = time.perf_counter()
        try:
            raw = np.asarray(self.session.run([self.output_name], self.queues)[0], dtype=np.float32)
        except Exception as error:
            self.reset()
            raise RuntimeError(f"Teacher inference failed: {error}") from error
        latency = (time.perf_counter() - start) * 1000
        if not np.isfinite(raw).all():
            raise RuntimeError("Teacher returned nonfinite outputs")
        action = raw[0, self.slices["action"]].copy()
        features = raw[0, self.slices["hidden_state"]].copy()
        queue = self.queues["features_buffer"]
        queue[:, :-1] = queue[:, 1:].copy()
        queue[:, -1] = features
        index = self.steps
        self.steps += 1
        result = {
            "action": action,
            "features": features,
            "valid": index >= self.warmup_frames,
            "latency_ms": latency,
            "frame": index,
        }
        if "plan" in self.slices:
            plan = raw[0, self.slices["plan"]]
            if len(plan) != 2 * 33 * 15:
                raise ValueError("unsupported plan MDN layout; no silent parser substitution")
            result["plan"] = plan[: 33 * 15].reshape(33, 15).copy()
        return result
