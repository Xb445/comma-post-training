import math
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    manifest: Path
    history: int = 4
    image_size: tuple[int, int] = (32, 64)
    stride: int = 1
    max_gap_s: float = 0.3
    domain_mix: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int = 64
    vision_channels: int = 16
    latent_channels: int = 4
    transformer_layers: int = 2
    heads: int = 4
    freeze_encoder: bool = False
    init_checkpoint: Path | None = None


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: Path
    max_steps: int = 20
    batch_size: int = 4
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    seed: int = 42
    device: str = "cpu"
    num_threads: int = 2
    num_workers: int = 0
    val_every: int = 10
    val_batches: int = 0
    gradient_clip: float = 1.0


@dataclass(frozen=True)
class LossConfig:
    bc_weight: float = 1.0
    kd_weight: float = 0.0
    plan_weight: float = 0.1


@dataclass(frozen=True)
class WorldConfig:
    anchor_offsets: tuple[int, ...] = (2, 3)
    plan_steps: int = 3
    sampling_steps: int = 8


@dataclass(frozen=True)
class RunConfig:
    task: str
    data: DataConfig
    training: TrainingConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    world: WorldConfig = field(default_factory=WorldConfig)

    def __post_init__(self):
        if self.task not in ("policy", "compressor", "world_model"):
            raise ValueError("task must be policy, compressor, or world_model")
        positive = [
            self.data.history,
            self.data.stride,
            *self.data.image_size,
            self.model.hidden_size,
            self.model.vision_channels,
            self.model.latent_channels,
            self.model.transformer_layers,
            self.model.heads,
            self.training.max_steps,
            self.training.batch_size,
            self.training.num_threads,
            self.training.val_every,
            self.world.plan_steps,
            self.world.sampling_steps,
        ]
        if any(type(v) is not int or v < 1 for v in positive):
            raise ValueError(
                "dimensions, batch sizes, intervals and steps must be positive integers"
            )
        if len(self.data.image_size) != 2 or any(v % 4 for v in self.data.image_size):
            raise ValueError("image_size must have two dimensions divisible by four")
        if self.model.hidden_size % self.model.heads:
            raise ValueError("hidden_size must be divisible by heads")
        if type(self.training.seed) is not int or not 0 <= self.training.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if type(self.training.num_workers) is not int or self.training.num_workers < 0:
            raise ValueError("num_workers must be nonnegative")
        if type(self.training.val_batches) is not int or self.training.val_batches < 0:
            raise ValueError("val_batches must be nonnegative; zero evaluates the full split")
        if self.training.device not in ("cpu", "cuda", "auto"):
            raise ValueError("device must be cpu, cuda, or auto")
        for name, value in (
            ("learning_rate", self.training.learning_rate),
            ("gradient_clip", self.training.gradient_clip),
            ("max_gap_s", self.data.max_gap_s),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        for value in (self.training.weight_decay, *asdict(self.loss).values()):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError("loss weights and weight_decay must be finite and nonnegative")
        if self.task == "policy" and self.loss.bc_weight + self.loss.kd_weight == 0:
            raise ValueError("policy requires BC or KD supervision")
        if not isinstance(self.data.domain_mix, dict) or any(
            key not in ("real", "sim", "synthetic")
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            for key, value in self.data.domain_mix.items()
        ):
            raise ValueError("domain_mix must map real/sim/synthetic to positive finite weights")
        if len(set(self.world.anchor_offsets)) != len(self.world.anchor_offsets) or any(
            type(v) is not int or v < 2 for v in self.world.anchor_offsets
        ):
            raise ValueError(
                "anchor_offsets must be unique integers >=2; target offset 1 is excluded"
            )
        if type(self.model.freeze_encoder) is not bool:
            raise ValueError("freeze_encoder must be boolean")
        if (
            self.model.freeze_encoder
            and self.task != "world_model"
            and self.model.init_checkpoint is None
        ):
            raise ValueError("freezing an encoder requires an initialized checkpoint")

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: Path) -> RunConfig:
    with path.open("rb") as stream:
        document = tomllib.load(stream)
    if set(document) - {"task", "data", "training", "model", "loss", "world"}:
        raise ValueError("unknown configuration section")
    sections = {}
    for name, cls in (
        ("data", DataConfig),
        ("training", TrainingConfig),
        ("model", ModelConfig),
        ("loss", LossConfig),
        ("world", WorldConfig),
    ):
        values = dict(document.get(name, {}))
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown {name} options: {sorted(unknown)}")
        for key in ("manifest", "output_dir", "init_checkpoint"):
            if key in values:
                if not isinstance(values[key], str) or not values[key].strip():
                    raise ValueError(f"{key} must be a nonempty path")
                values[key] = (path.parent / values[key]).resolve()
        for key in ("image_size", "anchor_offsets"):
            if key in values:
                values[key] = tuple(values[key])
        try:
            sections[name] = cls(**values)
        except TypeError as error:
            raise ValueError(f"invalid {name} configuration: {error}") from error
    return RunConfig(task=document.get("task", ""), **sections)
