"""Causal windows, compact valid-run indices, and reproducible per-step sampling."""

from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from comma_post_training.data_pipeline.episodes import Catalogue, Episode
from comma_post_training.data_pipeline.geometry import motion_vector, relative_pose


@dataclass(frozen=True)
class Span:
    episode: int
    first: int
    count: int
    run_start: int


class EpisodeWindows(Dataset):
    def __init__(
        self,
        catalogue: Catalogue,
        *,
        split: str,
        task: str,
        history: int = 4,
        image_size: tuple[int, int] = (32, 64),
        stride: int = 1,
        max_gap_s: float = 0.3,
        anchors: tuple[int, ...] = (2, 3),
        plan_steps: int = 3,
        supervision: tuple[str, ...] = ("expert",),
    ):
        self.catalogue, self.task, self.history = catalogue, task, history
        self.image_size, self.stride = image_size, stride
        self.anchors, self.plan_steps = anchors, plan_steps
        self.spans, self.ends = [], []
        self.cache: OrderedDict[int, Episode] = OrderedDict()
        self.domain_spans: dict[str, list[int]] = {}
        self.domain_ends: dict[str, list[int]] = {}
        self.encoder_fingerprint = None
        self.latent_shape = None
        self.teacher_identity = None
        self.teacher_feature_dim = None
        if task == "compressor":
            self.history = 1
        future = max(1, plan_steps, *anchors) if task == "world_model" else 0
        for number, reference in enumerate(catalogue.episodes):
            if reference.split != split:
                continue
            episode = Episode(reference)
            n = episode.validate()
            times = episode.array("timestamps")
            valid = episode.array("valid") if "valid" in episode.paths else np.ones(n, dtype=bool)
            target_valid = np.ones(n, dtype=bool)
            if task in ("policy", "official_residual"):
                target_valid[:] = False
                for source in supervision:
                    if source + "_actions" in episode.paths:
                        mask = source + "_valid"
                        target_valid |= episode.array(mask) if mask in episode.paths else True
            if task == "official_residual":
                features = episode.array("teacher_features")
                identity = episode.metadata["teacher_identity"]
                if self.teacher_identity is not None and identity != self.teacher_identity:
                    raise ValueError("residual windows cannot mix Teacher models or adapters")
                self.teacher_identity, self.teacher_feature_dim = identity, features.shape[1]
                target_valid &= episode.array("teacher_valid")
            if task == "world_model":
                latent = episode.array("latents")
                episode.array("poses")
                identity = episode.metadata["encoder_fingerprint"]
                shape = latent.shape[1:]
                if self.encoder_fingerprint is not None and (
                    identity != self.encoder_fingerprint or shape != self.latent_shape
                ):
                    raise ValueError("world-model windows cannot mix latent encoders or shapes")
                self.encoder_fingerprint, self.latent_shape = identity, shape
            start = None
            for i in range(n + 1):
                boundary = (
                    i == n
                    or not valid[i]
                    or (start is not None and i > start and times[i] - times[i - 1] > max_gap_s)
                )
                if boundary and start is not None:
                    centers = np.arange(start + self.history - 1, i - future, stride)
                    centers = centers[target_valid[centers]]
                    cuts = np.r_[0, np.flatnonzero(np.diff(centers) != stride) + 1, len(centers)]
                    for left, right in zip(cuts[:-1], cuts[1:], strict=True):
                        if right > left:
                            self._add_span(number, int(centers[left]), int(right - left), start)
                    start = None
                if i < n and valid[i] and start is None:
                    start = i
        if not self.spans:
            raise ValueError(
                f"no usable {task} windows in split={split}; check lengths/gaps/labels"
            )

    def _add_span(self, episode: int, first: int, count: int, run_start: int) -> None:
        domain = self.catalogue.episodes[episode].domain
        self.domain_spans.setdefault(domain, []).append(len(self.spans))
        domain_ends = self.domain_ends.setdefault(domain, [])
        domain_ends.append((domain_ends[-1] if domain_ends else 0) + count)
        self.spans.append(Span(episode, first, count, run_start))
        self.ends.append((self.ends[-1] if self.ends else 0) + count)

    def __len__(self) -> int:
        return self.ends[-1]

    def sample_index(self, rng: np.random.Generator, domain: str | None = None) -> int:
        if domain is None:
            return int(rng.integers(len(self)))
        ends = self.domain_ends[domain]
        draw = int(rng.integers(ends[-1]))
        part = bisect_right(ends, draw)
        span_index = self.domain_spans[domain][part]
        global_start = self.ends[span_index - 1] if span_index else 0
        return global_start + draw - (ends[part - 1] if part else 0)

    def label_counts(self, domains: set[str] | None = None) -> dict[str, int]:
        counts = {"expert": 0, "teacher": 0}
        for span in self.spans:
            ref = self.catalogue.episodes[span.episode]
            if domains is not None and ref.domain not in domains:
                continue
            episode = Episode(ref)
            indices = span.first + np.arange(span.count) * self.stride
            for source in counts:
                if source + "_actions" in episode.paths:
                    mask = source + "_valid"
                    counts[source] += (
                        int(episode.array(mask)[indices].sum())
                        if mask in episode.paths
                        else span.count
                    )
        return counts

    def __getitem__(self, index: int) -> dict:
        if not 0 <= index < len(self):
            raise IndexError(index)
        part = bisect_right(self.ends, index)
        span = self.spans[part]
        local = index - (self.ends[part - 1] if part else 0)
        t = span.first + local * self.stride
        if span.episode not in self.cache:
            self.cache[span.episode] = Episode(self.catalogue.episodes[span.episode])
            if len(self.cache) > 4:
                self.cache.popitem(last=False)
        self.cache.move_to_end(span.episode)
        episode = self.cache[span.episode]
        history = list(range(t - self.history + 1, t + 1))
        times = episode.array("timestamps")
        result = {
            "episode_id": episode.reference.id,
            "domain": episode.reference.domain,
            "frame": t,
        }
        if self.task in ("compressor", "policy"):
            result["images"] = torch.from_numpy(episode.images(history, self.image_size))
        if self.task in ("policy", "official_residual"):
            past = np.zeros((self.history, 2), dtype=np.float32)
            for i, frame in enumerate(history):
                if frame > span.run_start:
                    past[i] = episode.array("executed_actions")[frame - 1]
            result.update(
                state=torch.from_numpy(
                    np.array(episode.array("ego_state")[history], dtype=np.float32)
                ),
                past_actions=torch.from_numpy(past),
                dt=torch.tensor(
                    np.diff(times[history], prepend=times[history[0]]), dtype=torch.float32
                )[:, None],
            )
            for source in ("expert", "teacher"):
                label = source + "_actions"
                mask = source + "_valid"
                active = label in episode.paths and (
                    mask not in episode.paths or bool(episode.array(mask)[t])
                )
                values = episode.array(label)[t] if active else np.zeros(2)
                result[source] = torch.tensor(np.array(values), dtype=torch.float32)
                result[source + "_valid"] = torch.tensor(active)
            if self.task == "official_residual":
                result["teacher_features"] = torch.tensor(
                    np.array(episode.array("teacher_features")[t]), dtype=torch.float32
                )
        if self.task == "world_model":
            latents, poses = episode.array("latents"), episode.array("poses")
            anchor_indices = [t + offset for offset in self.anchors]
            for name, indices in (("history_latents", history), ("anchor_latents", anchor_indices)):
                value = np.array(latents[indices], dtype=np.float32)
                if not np.isfinite(value).all():
                    raise ValueError("nonfinite latent in selected window")
                result[name] = torch.from_numpy(value)
            target = np.array(latents[t + 1], dtype=np.float32)
            if not np.isfinite(target).all():
                raise ValueError("nonfinite target latent")
            result.update(
                target_latent=torch.from_numpy(target),
                motion=torch.from_numpy(motion_vector(poses[t], poses[t + 1])),
                frame_times=torch.tensor(
                    times[history + anchor_indices + [t + 1]] - times[t], dtype=torch.float32
                ),
                plan=torch.tensor(
                    np.stack(
                        [
                            relative_pose(poses[t], poses[t + i])[:2, 3]
                            for i in range(1, self.plan_steps + 1)
                        ]
                    ),
                    dtype=torch.float32,
                ),
                plan_times=torch.tensor(
                    times[t + 1 : t + self.plan_steps + 1] - times[t], dtype=torch.float32
                ),
            )
        return result


class StepBatchSampler(Sampler):
    def __init__(
        self,
        dataset: EpisodeWindows,
        *,
        seed: int,
        start: int,
        steps: int,
        batch_size: int,
        domain_mix: dict[str, float],
    ):
        self.dataset, self.seed, self.start, self.steps = dataset, seed, start, steps
        self.batch_size = batch_size
        self.domains = sorted(domain_mix)
        weights = np.asarray([domain_mix[domain] for domain in self.domains], dtype=np.float64)
        if self.domains and (
            any(domain not in dataset.domain_ends for domain in self.domains)
            or not np.isfinite(weights).all()
            or np.any(weights <= 0)
        ):
            raise ValueError(
                "domain_mix must assign positive weights to available training domains"
            )
        self.probabilities = weights / weights.sum() if self.domains else None

    def __iter__(self):
        for step in range(self.start, self.steps):
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, step]))
            domains = (
                rng.choice(self.domains, self.batch_size, p=self.probabilities)
                if self.domains
                else [None] * self.batch_size
            )
            yield [self.dataset.sample_index(rng, domain) for domain in domains]

    def __len__(self):
        return max(0, self.steps - self.start)
