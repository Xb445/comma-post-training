"""Stream a recordable simulator into versioned canonical episodes and diagnostic plots."""

import copy
import html
from pathlib import Path

import numpy as np
from PIL import Image

from comma_post_training.data_pipeline.episodes import Catalogue
from comma_post_training.data_pipeline.geometry import validate_poses
from comma_post_training.engine.checkpoints import atomic_json, staged_directory
from comma_post_training.simulator.api import (
    check_action,
    check_observation,
    check_transition,
    load_environment,
)
from comma_post_training.simulator.recording import DT, recording_spec

DEFAULT_FACTORY = "comma_post_training.simulator.recording:RoadEnvironment"


def _polyline(points, x_range, y_range, box, color):
    x0, y0, width, height = box
    xmin, xmax = x_range
    ymin, ymax = y_range
    xmax, ymax = max(xmax, xmin + 1e-9), max(ymax, ymin + 1e-9)
    coords = [
        f"{x0 + (x - xmin) / (xmax - xmin) * width:.2f},"
        f"{y0 + height - (y - ymin) / (ymax - ymin) * height:.2f}"
        for x, y in points
    ]
    return (
        f'<polyline fill="none" stroke="{color}" stroke-width="1.5" points="{" ".join(coords)}"/>'
    )


def _diagnostic_svg(path, title, times, poses, expert, achieved):
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="760" viewBox="0 0 1000 760">',
        '<rect width="1000" height="760" fill="white"/>',
    ]
    svg.append(f'<text x="30" y="25" font-size="18">{html.escape(title)}</text>')
    panels = [
        (
            "FRD trajectory: x forward (m), y right (m)",
            [(poses[:, 0, 3], poses[:, 1, 3], "#2463a6")],
        ),
        (
            "Lateral acceleration (m/s^2): expert blue, achieved orange",
            [(times, expert[:, 0], "#2463a6"), (times, achieved[:, 0], "#d36520")],
        ),
        (
            "Longitudinal acceleration (m/s^2): expert blue, achieved orange",
            [(times, expert[:, 1], "#2463a6"), (times, achieved[:, 1], "#d36520")],
        ),
    ]
    for index, (label, curves) in enumerate(panels):
        top = 60 + index * 230
        svg.append(f'<text x="50" y="{top - 10}" font-size="14">{label}</text>')
        svg.append(
            f'<rect x="60" y="{top}" width="900" height="180" fill="#fafafa" stroke="#999"/>'
        )
        xs = np.concatenate([curve[0] for curve in curves])
        ys = np.concatenate([curve[1] for curve in curves])
        bounds_x = (float(xs.min()), float(xs.max()))
        margin = max(float(np.ptp(ys)) * 0.1, 0.01)
        bounds_y = (float(ys.min() - margin), float(ys.max() + margin))
        for x, y, color in curves:
            svg.append(
                _polyline(zip(x, y, strict=True), bounds_x, bounds_y, (60, top, 900, 180), color)
            )
        svg.append(
            f'<text x="65" y="{top + 174}" font-size="11">'
            f"y range {bounds_y[0]:.3f} to {bounds_y[1]:.3f}</text>"
        )
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def record_simulation(
    output: Path,
    *,
    factory: str = DEFAULT_FACTORY,
    duration_s: float = 30.0,
    seed: int = 42,
    env_kwargs: dict | None = None,
    routes_per_split: dict[str, int] | None = None,
) -> dict:
    """Expert rollouts with label_t / achieved_interval_t causality and split isolation.

    A custom factory must implement Environment plus current_pose() and
    recording_metadata(); the latter declares actual camera calibration, never a
    guessed resize.  The backend must advance exactly 0.2 s per step.
    """
    if (
        not isinstance(duration_s, (int, float))
        or isinstance(duration_s, bool)
        or not np.isfinite(duration_s)
        or duration_s < 0.4
    ):
        raise ValueError("duration_s must be finite and at least 0.4 seconds")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    frames = int(round(duration_s / DT))
    if not np.isclose(frames * DT, duration_s, atol=1e-8):
        raise ValueError("duration_s must be an exact multiple of the 0.2 s sampling interval")
    groups = routes_per_split or {"train": 2, "val": 1, "test": 1}
    if set(groups) != {"train", "val", "test"} or any(
        type(v) is not int or v < 1 for v in groups.values()
    ):
        raise ValueError("routes_per_split requires positive train/val/test counts")
    if env_kwargs is not None and not isinstance(env_kwargs, dict):
        raise ValueError("env_kwargs must be a dictionary")
    spec = recording_spec()
    rows, scenarios, episode_reports = [], set(), []
    dataset_kind = None
    with staged_directory(output) as staging:
        for split, count in groups.items():
            for local in range(count):
                index = len(rows)
                route_seed = seed + index
                env = load_environment(factory, spec, split, copy.deepcopy(env_kwargs or {}))
                try:
                    if not callable(getattr(env, "current_pose", None)) or not callable(
                        getattr(env, "recording_metadata", None)
                    ):
                        raise ValueError(
                            "recording backend requires current_pose() and recording_metadata()"
                        )
                    observation = check_observation(env.reset(seed=route_seed), spec)
                    description = env.describe()
                    scenario = description.get("scenario_id")
                    route = description.get("route_id")
                    if not isinstance(scenario, str) or not scenario or scenario in scenarios:
                        raise ValueError(
                            "each recorded episode must identify a unique scenario; "
                            "split leakage is rejected"
                        )
                    if not isinstance(route, str) or not route:
                        raise ValueError("recording backend requires a real route_id")
                    scenarios.add(scenario)
                    current_kind = "synthetic" if description["kind"] == "synthetic" else "recorded"
                    if dataset_kind is not None and dataset_kind != current_kind:
                        raise ValueError(
                            "one recording dataset cannot mix synthetic fixture "
                            "and simulator evidence"
                        )
                    dataset_kind = current_kind
                    episode_id = f"sim-{split}-{local:03d}-{route_seed}"
                    folder = staging / "episodes" / episode_id
                    folder.mkdir(parents=True)
                    images = np.lib.format.open_memmap(
                        folder / "images.npy",
                        mode="w+",
                        dtype=np.uint8,
                        shape=(frames, 2, 256, 512, 3),
                    )
                    state, expert, executed, commanded = (
                        np.empty((frames, dimension), dtype=np.float32)
                        for dimension in (2, 2, 2, 2)
                    )
                    poses = np.empty((frames, 4, 4), dtype=np.float64)
                    rewards = np.empty(frames, dtype=np.float32)
                    terminated = np.zeros(frames, dtype=bool)
                    truncated = np.zeros(frames, dtype=bool)
                    initial_preview = None
                    for frame in range(frames):
                        rgb = np.rint(
                            observation["images"][0].numpy().transpose(0, 2, 3, 1) * 255
                        ).astype(np.uint8)
                        images[frame] = rgb
                        if frame == 0:
                            initial_preview = np.concatenate((rgb[0], rgb[1]), axis=1)
                        state[frame] = observation["state"][0].numpy()
                        poses[frame] = np.asarray(env.current_pose(), dtype=np.float64)
                        expert[frame] = check_action(env.expert_action())
                        commanded[frame] = expert[frame]
                        transition = check_transition(env.step(commanded[frame].copy()), spec)
                        if not np.isclose(
                            float(transition.observation["dt"][-1, 0]), DT, atol=1e-6
                        ):
                            raise ValueError(
                                "recording backend must step exactly 0.2 seconds / 5 Hz"
                            )
                        if not np.allclose(
                            transition.observation["past_actions"][-1].numpy(),
                            transition.achieved_action,
                            atol=1e-6,
                        ):
                            raise ValueError(
                                "next observation past_actions must record "
                                "this interval's achieved action"
                            )
                        executed[frame] = transition.achieved_action
                        rewards[frame] = transition.reward
                        terminated[frame], truncated[frame] = (
                            transition.terminated,
                            transition.truncated,
                        )
                        if frame + 1 < frames and (transition.terminated or transition.truncated):
                            raise ValueError(
                                "simulator ended before requested recording length; "
                                "no padding or silent episode concatenation"
                            )
                        observation = transition.observation
                    # The collector's time limit is a truncation, never a true terminal.
                    collector_truncated = not bool(terminated[-1] or truncated[-1])
                    if collector_truncated:
                        truncated[-1] = True
                    np.savez(
                        folder / "final-observation.npz",
                        images=np.rint(
                            observation["images"][0].numpy().transpose(0, 2, 3, 1) * 255
                        ).astype(np.uint8),
                        ego_state=observation["state"][0].numpy(),
                        past_actions=observation["past_actions"][0].numpy(),
                        pose=np.asarray(env.current_pose()),
                        timestamp_s=frames * DT,
                    )
                    validate_poses(poses)
                    images.flush()
                    del images
                    arrays = {
                        "timestamps": np.arange(frames, dtype=np.float64) * DT,
                        "ego_state": state,
                        "expert_actions": expert,
                        "executed_actions": executed,
                        "commanded_actions": commanded,
                        "poses": poses,
                        "valid": np.ones(frames, dtype=bool),
                        "expert_valid": np.ones(frames, dtype=bool),
                        "rewards": rewards,
                        "terminated": terminated,
                        "truncated": truncated,
                    }
                    for name, values in arrays.items():
                        np.save(folder / f"{name}.npy", values, allow_pickle=False)
                    identity = {
                        "id": episode_id,
                        "route_id": route,
                        "domain": "sim",
                        "split": split,
                    }
                    metadata = {
                        "format": "comma-episode-v1",
                        "episode": identity,
                        "scenario_id": scenario,
                        "timestamp_units": "s",
                        "coordinate_frame": "vehicle_frd",
                        "action_order": ["a_lat", "a_long"],
                        "action_units": "m/s^2",
                        "camera_names": spec["camera_names"],
                        "state_features": spec["state_features"],
                        "preprocessing_id": description.get(
                            "preprocessing_id", spec["preprocessing_id"]
                        ),
                        "arrays": {name: f"{name}.npy" for name in ["images", *arrays]},
                        "simulation": {
                            "factory": factory,
                            "seed": route_seed,
                            "description": description,
                            "sample_hz": 5,
                            "action_alignment": (
                                "label_t and achieved_interval_t_after_observation_t"
                            ),
                            "collector_truncated": collector_truncated,
                            "final_observation": "final-observation.npz",
                        },
                        **env.recording_metadata(),
                    }
                    if not isinstance(metadata.get("teacher_rgb"), dict):
                        raise ValueError(
                            "recording backend must declare its explicit teacher_rgb calibration"
                        )
                    atomic_json(folder / "episode.json", metadata)
                    Image.fromarray(initial_preview).save(folder / "camera-preview.png")
                    _diagnostic_svg(
                        folder / "diagnostics.svg",
                        episode_id,
                        arrays["timestamps"],
                        poses,
                        expert,
                        executed,
                    )
                    rows.append(
                        {
                            **identity,
                            "metadata": str((folder / "episode.json").relative_to(staging)),
                        }
                    )
                    episode_reports.append(
                        {
                            "id": episode_id,
                            "scenario_id": scenario,
                            "route_id": route,
                            "split": split,
                            "frames": frames,
                            "seed": route_seed,
                        }
                    )
                finally:
                    env.close()
        atomic_json(
            staging / "manifest.json",
            {
                "format": "comma-episodes-v1",
                "kind": dataset_kind,
                "revision": f"sim-recording-v1-seed-{seed}",
                "episodes": rows,
            },
        )
        quality = Catalogue(staging / "manifest.json").validate()
        report = {
            "manifest": str(output / "manifest.json"),
            "factory": factory,
            "duration_s": duration_s,
            "sample_hz": 5,
            "dataset_kind": dataset_kind,
            "episodes": episode_reports,
            "quality": quality,
            "pilot_ready": duration_s >= 30 and frames > 25 and groups["train"] >= 2,
            "evidence_scope": (
                "recording_contract_and_simulator_data_not_official_teacher_or_driving_quality"
            ),
        }
        atomic_json(staging / "recording-report.json", report)
    # Fingerprint paths are recomputed after atomic relocation, not copied from staging.
    report["quality"] = Catalogue(output / "manifest.json").validate()
    atomic_json(output / "recording-report.json", report)
    return report
