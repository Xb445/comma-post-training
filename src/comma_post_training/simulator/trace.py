"""Optional evaluation evidence; captures observed policy rollouts without rerendering them."""

from contextlib import contextmanager

import numpy as np
from PIL import Image

from comma_post_training.engine.checkpoints import atomic_json, staged_directory


class EvaluationTrace:
    def __init__(self, root, spec, split, policy_name):
        self.root, self.spec = root, spec
        self.document = {
            "format": "comma-evaluation-trace-v1",
            "split": split,
            "policy": policy_name,
            "model_spec": spec,
            "episodes": [],
            "image_semantics": "pre_action_observation_rgb_jpeg_for_display",
            "action_order": ["a_lat", "a_long"],
            "action_units": "m/s^2",
            "coordinate_frame": "vehicle_frd",
        }

    def start(self, number, seed, provenance):
        self.folder = self.root / f"episode-{number:03d}"
        self.folder.mkdir()
        self.rows = []
        self.current = {"id": f"evaluation-{number:03d}", "seed": seed, **provenance}

    def append(
        self,
        observation,
        *,
        timestamp_s,
        pose,
        next_pose,
        expert,
        official,
        command,
        transition,
        warmed,
        truncated,
        diagnostics,
    ):
        number = len(self.rows)
        images = np.rint(observation["images"][-1].numpy().transpose(0, 2, 3, 1) * 255).astype(
            np.uint8
        )
        paths = []
        for camera, pixels in enumerate(images):
            path = self.folder / f"{number:05d}-{camera}.jpg"
            Image.fromarray(pixels).save(path, quality=90)
            paths.append(str(path.relative_to(self.root)))
        self.rows.append(
            {
                "frame": number,
                "timestamp_s": timestamp_s,
                "interval_s": float(transition.observation["dt"][-1, 0]),
                "images": paths,
                "state": observation["state"][-1].tolist(),
                "pose": pose,
                "next_pose": next_pose,
                "expert_action": expert,
                "official_action": official,
                "commanded_action": command,
                "achieved_action": transition.achieved_action.tolist(),
                "teacher_warmed": warmed,
                "reward": transition.reward,
                "terminated": transition.terminated,
                "truncated": truncated,
                "diagnostics_after_step": diagnostics,
            }
        )

    def end(self):
        path = self.folder / "steps.json"
        atomic_json(path, {"steps": self.rows})
        self.document["episodes"].append(
            {
                **self.current,
                "steps": str(path.relative_to(self.root)),
                "frames": len(self.rows),
            }
        )

    def finish(self, metrics):
        atomic_json(self.root / "metrics.json", metrics)
        atomic_json(self.root / "trace.json", self.document)


@contextmanager
def evaluation_trace(output, spec, split, policy_name):
    if output is None:
        yield None
        return
    with staged_directory(output) as staging:
        yield EvaluationTrace(staging, spec, split, policy_name)


def pose_of(env):
    return np.asarray(env.current_pose()).tolist() if hasattr(env, "current_pose") else None
