"""Build a portable HTML report plus PNG/SVG, CSV, optional GIF and JSON evidence."""

import base64
import csv
import html
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from comma_post_training.data_pipeline.episodes import Catalogue, Episode
from comma_post_training.engine.checkpoints import atomic_json, file_sha256, staged_directory
from comma_post_training.reporting.inputs import (
    clean,
    compare_runs,
    episode_values,
    evaluation_summary,
    load_predictions,
    load_run,
    read_json,
)


def _samples(length, maximum):
    return np.unique(np.linspace(0, length - 1, min(length, maximum)).astype(int))


def _image(episode, frame, view):
    if "images" in episode.paths:
        return Image.fromarray(np.asarray(episode.array("images")[frame, view]))
    index, paths = episode.image_paths
    with Image.open(index.parent / paths[frame][view]) as image:
        return image.convert("RGB")


def _encode(image, width):
    image = image.copy()
    image.thumbnail((width, width), Image.Resampling.LANCZOS)
    stream = io.BytesIO()
    image.save(stream, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode(), image


def _csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(clean(value), ensure_ascii=False)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def _frame_rows(values):
    for frame, time in enumerate(values["time_s"]):
        row = {"frame": frame, "time_s": float(time)}
        for name in ("expert", "teacher", "adapted", "commanded", "executed"):
            if name in values:
                row.update(
                    {
                        name + "_" + axis: clean(values[name][frame, i])
                        for i, axis in enumerate(("a_lat_mps2", "a_long_mps2"))
                    }
                )
        for name in ("teacher_valid", "valid", "rewards", "terminated", "truncated"):
            if name in values:
                row[name] = clean(values[name][frame])
        if "xy" in values:
            row["world_x_m"], row["world_y_m"] = values["xy"][frame]
        for i, name in enumerate(values["state_features"]):
            row[name] = values["state"][frame, i]
        yield row


def _viewer(name, source, values, camera_names, indices, images):
    return {
        "name": name,
        "source": source,
        "camera_names": camera_names,
        "original_frames": len(values["time_s"]),
        "indices": indices.tolist(),
        "values": clean(
            {
                key: np.asarray(value)[indices] if key != "state_features" else value
                for key, value in values.items()
            }
        ),
        "images": images,
    }


def _gif(path, frames, times):
    size = (max(frame.width for frame in frames), max(frame.height for frame in frames))
    normalized = []
    for frame in frames:
        canvas = Image.new("RGB", size, "#15212d")
        canvas.paste(frame, (0, 0))
        normalized.append(canvas)
    durations = np.diff(times) * 1000
    durations = np.r_[durations, np.median(durations) if len(durations) else 200]
    normalized[0].save(
        path,
        save_all=True,
        append_images=normalized[1:],
        loop=0,
        duration=np.maximum(20, np.rint(durations)).astype(int).tolist(),
        optimize=False,
    )


def _data_viewers(
    catalogue, references, predictions, figures, root, max_episodes, max_frames, width, gif
):
    viewers = []
    for number, ref in enumerate(catalogue.episodes[:max_episodes]):
        episode = references[ref.id]
        values = episode_values(episode)
        values["state_features"] = episode.metadata["state_features"]
        matches = [
            [row for row in pred["rows"] if row["episode_id"] == ref.id] for pred in predictions
        ]
        matches = [rows for rows in matches if rows]
        if len(matches) == 1:
            adapted = np.full((len(values["time_s"]), 2), np.nan)
            teacher = np.full_like(adapted, np.nan)
            teacher_valid = np.zeros(len(adapted), dtype=bool)
            for row in matches[0]:
                adapted[row["frame"]] = row["adapted_action"]
                teacher[row["frame"]] = row["teacher_action"]
                teacher_valid[row["frame"]] = True
            values["adapted"] = adapted
            if "teacher" not in values:
                values["teacher"], values["teacher_valid"] = teacher, teacher_valid
        name = f"episode-{number:03d}"
        figures.episode(values, name, f"{ref.id} · {ref.split} · {catalogue.kind}")
        _csv(root / f"{name}.csv", _frame_rows(values))
        indices = _samples(len(values["time_s"]), max_frames)
        images, frames = [], []
        for frame in indices:
            encoded = [
                _encode(_image(episode, frame, view), width)
                for view in range(len(episode.metadata["camera_names"]))
            ]
            images.append([item[0] for item in encoded])
            if gif:
                canvas = Image.new(
                    "RGB",
                    (width * len(encoded), max(image.height for _, image in encoded) + 24),
                    "#15212d",
                )
                draw = ImageDraw.Draw(canvas)
                for view, (_, image) in enumerate(encoded):
                    canvas.paste(image, (view * width, 24))
                draw.text((5, 4), f"{ref.id}  t={values['time_s'][frame]:.2f}s", fill="white")
                frames.append(canvas)
        if gif:
            _gif(root / f"{name}.gif", frames, values["time_s"][indices])
        viewers.append(
            _viewer(
                ref.id,
                f"recorded episode · {ref.split} · {catalogue.kind}",
                values,
                episode.metadata["camera_names"],
                indices,
                images,
            )
        )
    return viewers


def _trace_viewers(paths, figures, root, max_episodes, max_frames, width):
    viewers, summaries = [], []
    for group, path in enumerate(paths):
        trace = read_json(path / "trace.json")
        if trace.get("format") != "comma-evaluation-trace-v1":
            raise ValueError(f"{path}: unsupported evaluation trace")
        summaries.append(
            {
                "path": str(path),
                "policy": trace["policy"],
                "split": trace["split"],
                "metrics": read_json(path / "metrics.json"),
            }
        )
        summaries[-1]["evaluation"] = evaluation_summary(summaries[-1]["metrics"])
        for number, episode in enumerate(trace["episodes"][:max_episodes]):
            rows = read_json(path / episode["steps"])["steps"]
            if not rows:
                raise ValueError(f"{path}: empty evaluation trace")
            values = {
                "time_s": np.array([row["timestamp_s"] for row in rows]),
                "state": np.array([row["state"] for row in rows]),
                "state_features": trace["model_spec"]["state_features"],
                "teacher_valid": np.array([row["teacher_warmed"] for row in rows]),
            }
            for key, source in (
                ("expert", "expert_action"),
                ("teacher", "official_action"),
                ("commanded", "commanded_action"),
                ("executed", "achieved_action"),
                ("rewards", "reward"),
                ("terminated", "terminated"),
                ("truncated", "truncated"),
            ):
                values[key] = np.array([row[source] for row in rows])
            values["teacher"] = values["teacher"].astype(float)
            values["teacher"][~values["teacher_valid"]] = np.nan
            if all(row.get("pose") is not None for row in rows):
                values["xy"] = np.array([row["pose"] for row in rows])[:, :2, 3]
            if all(isinstance(row.get("diagnostics_after_step"), dict) for row in rows):
                diagnostics = [row["diagnostics_after_step"] for row in rows]
                for key, target in (
                    ("cross_track_m", "cross_track_after_m"),
                    ("road_half_width_m", "road_half_width_m"),
                ):
                    if all(key in row for row in diagnostics):
                        values[target] = np.array([row[key] for row in diagnostics])
                values["interval_s"] = np.array([row["interval_s"] for row in rows])
            name = f"trace-{group:02d}-{number:03d}"
            figures.episode(
                values, name, f"{path.name} · {episode['scenario_id']} · {trace['split']}"
            )
            figures.trace_diagnostics(values, name + "-diagnostics")
            _csv(root / f"{name}.csv", rows)
            indices = _samples(len(rows), max_frames)
            images = []
            for frame in indices:
                current = []
                for filename in rows[frame]["images"]:
                    with Image.open(path / filename) as image:
                        current.append(_encode(image.convert("RGB"), width)[0])
                images.append(current)
            viewers.append(
                _viewer(
                    path.name + " / " + episode["id"],
                    f"policy rollout · {trace['policy']} · {trace['split']}",
                    values,
                    trace["model_spec"]["camera_names"],
                    indices,
                    images,
                )
            )
    return viewers, summaries


def build_report(
    output: Path,
    *,
    manifest: Path | None = None,
    run_dirs=(),
    prediction_dirs=(),
    smoke: Path | None = None,
    trace_dirs=(),
    max_episodes=4,
    max_frames=150,
    image_width=384,
    gif=False,
):
    for name, value, minimum in (
        ("max_episodes", max_episodes, 1),
        ("max_frames", max_frames, 2),
        ("image_width", image_width, 32),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >={minimum}")
    if not (manifest or run_dirs or prediction_dirs or smoke or trace_dirs):
        raise ValueError("provide at least one dataset, run, prediction, smoke or evaluation trace")
    from comma_post_training.reporting.plots import Figures

    references, quality, catalogue = None, None, None
    if manifest:
        catalogue = Catalogue(manifest)
        quality = catalogue.validate()
        references = {ref.id: Episode(ref) for ref in catalogue.episodes}
    runs = [load_run(path) for path in run_dirs]
    predictions = [load_predictions(path, references) for path in prediction_dirs]
    smoke_report = read_json(smoke) if smoke else None
    with staged_directory(output) as staging:
        figures = Figures(staging / "figures")
        viewers = (
            _data_viewers(
                catalogue,
                references,
                predictions,
                figures,
                staging,
                max_episodes,
                max_frames,
                image_width,
                gif,
            )
            if catalogue
            else []
        )
        traced, trace_summary = _trace_viewers(
            trace_dirs, figures, staging, max_episodes, max_frames, image_width
        )
        viewers.extend(traced)
        for number, run in enumerate(runs):
            figures.training(run, f"training-{number:02d}")
            _csv(staging / f"events-{number:02d}.csv", run["events"])
            if "evaluations" in run["summary"]:
                figures.closed_loop(run, f"closed-loop-{number:02d}")
                figures.online_actions(run, f"online-actions-{number:02d}")
        for number, prediction in enumerate(predictions):
            figures.prediction(prediction, f"prediction-{number:02d}")
            _csv(staging / f"predictions-{number:02d}.csv", prediction["rows"])
        inputs = [
            path
            for path in [
                manifest,
                smoke,
                *[p / "metrics.json" for p in run_dirs],
                *[
                    p / name
                    for p in run_dirs
                    for name in ("run.json", "events.jsonl")
                    if (p / name).exists()
                ],
                *[p / "predictions.jsonl" for p in prediction_dirs],
                *[
                    p / name
                    for p in prediction_dirs
                    for name in ("metrics.json", "gate2.json")
                    if (p / name).exists()
                ],
                *[p / "trace.json" for p in trace_dirs],
            ]
            if path
        ]
        summary = {
            "format": "comma-visual-report-v1",
            "dataset": quality,
            "runs": [run["summary"] for run in runs],
            "comparison": compare_runs(runs),
            "predictions": [
                {key: value for key, value in pred.items() if key != "rows"} for pred in predictions
            ],
            "teacher_smoke": smoke_report,
            "evaluation_traces": trace_summary,
            "inputs": [
                {"path": str(path.resolve()), "sha256": file_sha256(path)} for path in inputs
            ],
            "viewer": {
                "episodes_shown": len(viewers),
                "max_episodes_per_source": max_episodes,
                "max_frames_per_episode": max_frames,
                "images_are_display_thumbnails": True,
            },
            "scope": (
                "observed logs; single seed and synthetic simulation "
                "do not establish driving generalization"
            ),
        }
        atomic_json(staging / "summary.json", clean(summary))
        gallery = []
        for item in figures.items:
            png, svg = item["paths"]
            uri = "data:image/png;base64," + base64.b64encode(png.read_bytes()).decode()
            gallery.append(
                f'<article class="figure"><h3>{html.escape(item["title"])}</h3>'
                f'<img loading="lazy" src="{uri}" alt="{html.escape(item["title"])}">'
                f"<p>{html.escape(item['caption'])}</p>"
                f'<a href="figures/{png.name}" download>PNG</a> · '
                f'<a href="figures/{svg.name}" download>SVG</a></article>'
            )
        payload = json.dumps(
            clean({"summary": summary, "viewers": viewers}), ensure_ascii=False, allow_nan=False
        )
        payload = payload.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
        template = Path(__file__).with_name("template.html").read_text(encoding="utf-8")
        template = template.replace("__FIGURES__", "\n".join(gallery)).replace("__DATA__", payload)
        (staging / "index.html").write_text(template, encoding="utf-8")
    return {
        "report": str(output / "index.html"),
        "summary": str(output / "summary.json"),
        "figures": len(figures.items),
        "viewer_episodes": len(viewers),
        "read_only_inputs": True,
    }
