"""Config-driven training with held-out validation and reproducible resumable batches."""

import hashlib
import json
import math
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from comma_post_training.data_pipeline.episodes import Catalogue
from comma_post_training.data_pipeline.windows import EpisodeWindows, StepBatchSampler
from comma_post_training.engine.checkpoints import (
    atomic_json,
    file_sha256,
    load_checkpoint,
    save_checkpoint,
    source_fingerprint,
)
from comma_post_training.engine.config import RunConfig
from comma_post_training.engine.tasks import (
    architecture,
    build_model,
    build_spec,
    metric_objective,
    task_loss,
)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA requested but unavailable; this project's default lock installs CPU Torch"
        )
    return torch.device(requested)


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in batch.items()
    }


def make_windows(
    config: RunConfig, catalogue: Catalogue, split: str, *, evaluation: bool = False
) -> EpisodeWindows:
    supervision = (
        ("expert", "teacher")
        if evaluation
        else tuple(
            name
            for name, weight in (
                ("expert", config.loss.bc_weight),
                ("teacher", config.loss.kd_weight),
            )
            if weight > 0
        )
    )
    return EpisodeWindows(
        catalogue,
        split=split,
        task=config.task,
        history=config.data.history,
        image_size=config.data.image_size,
        stride=config.data.stride,
        max_gap_s=config.data.max_gap_s,
        anchors=config.world.anchor_offsets,
        plan_steps=config.world.plan_steps,
        supervision=supervision,
    )


class Metrics:
    def __init__(self):
        self.sums, self.counts = Counter(), Counter()
        self.domain_sums, self.domain_counts = defaultdict(Counter), defaultdict(Counter)

    def add(self, values: dict, domains: list[str]):
        for name, (value, mask) in values.items():
            array = value.detach().cpu().numpy()
            valid = mask.detach().cpu().numpy()
            if array.shape != valid.shape or len(array) != len(domains):
                raise ValueError("metric values/masks must align with batch domains")
            if not np.isfinite(array[valid]).all():
                raise RuntimeError(f"nonfinite metric {name}")
            self.sums[name] += float(array[valid].sum(dtype=np.float64))
            self.counts[name] += int(valid.sum())
            for domain in set(domains):
                selection = valid & (np.asarray(domains) == domain)
                self.domain_sums[domain][name] += float(array[selection].sum(dtype=np.float64))
                self.domain_counts[domain][name] += int(selection.sum())

    def report(self) -> dict:
        def average(sums, counts):
            return {key: sums[key] / count for key, count in counts.items() if count}

        return {
            "metrics": average(self.sums, self.counts),
            "counts": dict(self.counts),
            "domains": {
                domain: {
                    "metrics": average(self.domain_sums[domain], self.domain_counts[domain]),
                    "counts": dict(self.domain_counts[domain]),
                }
                for domain in sorted(self.domain_sums)
            },
        }


@torch.no_grad()
def validate(
    model,
    dataset: EpisodeWindows,
    config: RunConfig,
    device: torch.device,
    *,
    limit_batches: int = 0,
    require_objective: bool = True,
) -> dict:
    meter, windows = Metrics(), 0
    model.eval()
    loader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=False)
    devices = [device.index or 0] if device.type == "cuda" else []
    # A fixed validation draw enables comparison without consuming training RNG state.
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config.training.seed + 1009)
        for number, batch in enumerate(loader):
            if limit_batches and number >= limit_batches:
                break
            _, values = task_loss(model, move_batch(batch, device), config)
            meter.add(values, batch["domain"])
            windows += len(batch["domain"])
    report = meter.report()
    try:
        objective = metric_objective(report["metrics"], config)
    except ValueError:
        if require_objective:
            raise
        objective = None
    report.update(
        loss=objective,
        windows=windows,
        available_windows=len(dataset),
        full_split=windows == len(dataset),
    )
    return report


def _resume_signature(config: RunConfig, spec: dict, fingerprint: str) -> str:
    document = asdict(config)
    for name in ("output_dir", "max_steps"):
        document["training"].pop(name)
    payload = {
        "config": document,
        "model_spec": spec,
        "data": fingerprint,
        "source": source_fingerprint(),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "resolved_device": str(select_device(config.training.device)),
        "init_sha256": file_sha256(config.model.init_checkpoint)
        if config.model.init_checkpoint
        else None,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def train(config: RunConfig, *, resume: Path | None = None) -> dict:
    device = select_device(config.training.device)
    torch.set_num_threads(config.training.num_threads)
    torch.manual_seed(config.training.seed)
    catalogue = Catalogue(config.data.manifest)
    quality = catalogue.validate()
    training = make_windows(config, catalogue, "train")
    validation = make_windows(config, catalogue, "val")
    if config.task == "policy":
        counts = training.label_counts(set(config.data.domain_mix) or None)
        for source, weight in (
            ("expert", config.loss.bc_weight),
            ("teacher", config.loss.kd_weight),
        ):
            if weight and not counts[source]:
                raise ValueError(f"no active training labels for {source} in sampled domains")
    if config.task == "world_model" and (
        training.latent_shape != validation.latent_shape
        or training.encoder_fingerprint != validation.encoder_fingerprint
    ):
        raise ValueError("train/val latent encoders must be identical")
    spec = build_spec(config, quality["signature"], training)
    signature = _resume_signature(config, spec, catalogue.fingerprint)
    output = config.training.output_dir
    previous = load_checkpoint(resume) if resume is not None else None
    if output.exists() and any(output.iterdir()):
        marker = output / "run.json"
        if (
            previous is None
            or not marker.is_file()
            or json.loads(marker.read_text()).get("run_id") != previous["run_id"]
        ):
            raise ValueError("output directory is not empty or does not belong to this resumed run")
    model = build_model(spec)
    if config.model.init_checkpoint is not None:
        initialized = load_checkpoint(config.model.init_checkpoint, inference=True)
        if architecture(initialized["model_spec"]) != architecture(spec):
            raise ValueError("initial checkpoint architecture/data contract differs from this run")
        model.load_state_dict(initialized["model_state"], strict=True)
    if previous is not None:
        if previous["resume_signature"] != signature:
            raise ValueError(
                "resume contract changed: data/config/source differs; "
                "use init_checkpoint for fine-tuning"
            )
        model.load_state_dict(previous["model_state"], strict=True)
    if config.model.freeze_encoder and hasattr(model, "encoder"):
        model.encoder.requires_grad_(False)
    model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    start = previous["step"] if previous else 0
    if start >= config.training.max_steps:
        raise ValueError("max_steps must exceed the checkpoint step")
    if previous is not None:
        optimizer.load_state_dict(previous["optimizer_state"])
    sampler = StepBatchSampler(
        training,
        seed=config.training.seed,
        start=start,
        steps=config.training.max_steps,
        batch_size=config.training.batch_size,
        domain_mix=config.data.domain_mix,
    )
    run_id = previous["run_id"] if previous else uuid.uuid4().hex
    initial = validate(model, validation, config, device, limit_batches=config.training.val_batches)
    if previous is not None and (output / "last.pt").exists():
        if load_checkpoint(output / "last.pt")["step"] > previous["step"]:
            raise ValueError(
                "resume checkpoint is older than output last.pt; choose a new output_dir"
            )
    output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "run_id": run_id,
        "task": config.task,
        "model_spec": spec,
        "data": quality,
        "native_source_sha256": source_fingerprint(),
        "torch": torch.__version__,
        "device": str(device),
        "config": asdict(config),
        "resume_from": str(resume) if resume else None,
        "init_checkpoint_sha256": file_sha256(config.model.init_checkpoint)
        if config.model.init_checkpoint
        else None,
    }
    atomic_json(output / "run.json", provenance)
    best = (
        previous["best_validation_loss"]
        if previous and (output / "best.pt").exists()
        else initial["loss"]
    )
    loader = DataLoader(
        training,
        batch_sampler=sampler,
        num_workers=config.training.num_workers,
        generator=torch.Generator().manual_seed(config.training.seed + 17),
    )
    if previous is not None:
        torch.set_rng_state(previous["rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(previous["cuda_rng_state"])

    def checkpoint_payload(step, validation):
        return {
            "format": "comma-trainer-v1",
            "run_id": run_id,
            "task": config.task,
            "step": step,
            "model_spec": spec,
            "config": json.loads(json.dumps(asdict(config), default=str)),
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer_state": optimizer.state_dict(),
            "resume_signature": signature,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "best_validation_loss": best,
            "validation": validation,
            "encoder_fingerprint": model.encoder_fingerprint()
            if hasattr(model, "encoder_fingerprint")
            else None,
        }

    if not (output / "best.pt").exists():
        baseline = checkpoint_payload(start, initial)
        save_checkpoint(output / "initial.pt", baseline)
        save_checkpoint(output / "best.pt", baseline)
        save_checkpoint(output / "last.pt", baseline)
    last_validation = initial
    for step, batch in enumerate(loader, start=start + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, values = task_loss(model, move_batch(batch, device), config)
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite loss at step {step}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.training.gradient_clip, error_if_nonfinite=True
        )
        optimizer.step()
        event = {"step": step, "training_loss": float(loss.detach()), "gradient_norm": float(norm)}
        if step % config.training.val_every == 0 or step == config.training.max_steps:
            last_validation = validate(
                model, validation, config, device, limit_batches=config.training.val_batches
            )
            improved = last_validation["loss"] < best
            best = min(best, last_validation["loss"])
            checkpoint = checkpoint_payload(step, last_validation)
            save_checkpoint(output / "last.pt", checkpoint)
            if improved or not (output / "best.pt").exists():
                save_checkpoint(output / "best.pt", checkpoint)
            event["validation"] = last_validation
        with (output / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
    if not math.isfinite(last_validation["loss"]):
        raise RuntimeError("validation objective is not finite")
    report = {
        "task": config.task,
        "steps": config.training.max_steps,
        "train_windows": len(training),
        "initial_validation": initial,
        "final_validation": last_validation,
        "checkpoint": str(output / "last.pt"),
        "source_domains": sorted(training.domain_ends),
    }
    atomic_json(output / "metrics.json", report)
    return report


def evaluate(
    checkpoint: Path,
    *,
    manifest: Path | None = None,
    split: str = "test",
    device: str | None = None,
) -> dict:
    from dataclasses import replace

    saved = load_checkpoint(checkpoint, inference=True)
    # Rehydrate the stored configuration through its normal parser for identical validation.
    config = config_from_dict(saved["config"])
    if manifest is not None:
        config = replace(config, data=replace(config.data, manifest=manifest.resolve()))
    if device is not None:
        config = replace(config, training=replace(config.training, device=device))
    device = select_device(config.training.device)
    torch.set_num_threads(config.training.num_threads)
    catalogue = Catalogue(config.data.manifest)
    quality = catalogue.validate()
    dataset = make_windows(config, catalogue, split, evaluation=True)
    spec = build_spec(config, quality["signature"], dataset)
    if spec != saved["model_spec"]:
        raise ValueError("evaluation input/latent contract differs from checkpoint")
    model = build_model(spec).to(device)
    model.load_state_dict(saved["model_state"], strict=True)
    return {
        "task": config.task,
        "split": split,
        "data_fingerprint": catalogue.fingerprint,
        "objective_weights": asdict(config.loss),
        "loss_note": (
            "loss is null when the split lacks a label source required by the training objective"
        ),
        **validate(model, dataset, config, device, require_objective=False),
    }


def config_from_dict(document: dict) -> RunConfig:
    from comma_post_training.engine.config import (
        DataConfig,
        LossConfig,
        ModelConfig,
        TrainingConfig,
        WorldConfig,
    )

    data, training, model = (
        dict(document["data"]),
        dict(document["training"]),
        dict(document["model"]),
    )
    data["manifest"], training["output_dir"] = Path(data["manifest"]), Path(training["output_dir"])
    data["image_size"] = tuple(data["image_size"])
    if model["init_checkpoint"]:
        model["init_checkpoint"] = Path(model["init_checkpoint"])
    world = dict(document["world"])
    world["anchor_offsets"] = tuple(world["anchor_offsets"])
    return RunConfig(
        document["task"],
        DataConfig(**data),
        TrainingConfig(**training),
        ModelConfig(**model),
        LossConfig(**document["loss"]),
        WorldConfig(**world),
    )
