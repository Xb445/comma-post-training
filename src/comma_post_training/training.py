"""Small, explicit SFT loop; no model downloads or external experiment service."""

import json
import math
import tomllib
from dataclasses import asdict, dataclass, fields
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel

from comma_post_training.data import load_chats
from comma_post_training.tokenization import encode_chat, make_tokenizer


@dataclass(frozen=True)
class TrainConfig:
    data_path: Path
    output_dir: Path
    seed: int = 42
    max_steps: int = 20
    batch_size: int = 2
    learning_rate: float = 0.003
    max_length: int = 256
    hidden_size: int = 64
    num_layers: int = 2
    num_heads: int = 2
    num_threads: int = 2

    def __post_init__(self) -> None:
        for name in (
            "max_steps",
            "batch_size",
            "max_length",
            "hidden_size",
            "num_layers",
            "num_heads",
            "num_threads",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (int, float))
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError("learning_rate must be positive and finite")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")


def load_config(path: Path, output_dir: Path | None = None) -> TrainConfig:
    with path.open("rb") as stream:
        values = tomllib.load(stream)
    unknown = set(values) - {field.name for field in fields(TrainConfig)}
    if unknown:
        raise ValueError(f"unknown configuration fields: {', '.join(sorted(unknown))}")
    for name in ("data_path", "output_dir"):
        if not isinstance(values.get(name), str) or not values[name].strip():
            raise ValueError(f"{name} must be a nonempty path string")
        values[name] = (path.parent / values[name]).resolve()
    if output_dir is not None:
        values["output_dir"] = output_dir.resolve()
    return TrainConfig(**values)


def collate_batch(examples: list[dict[str, list[int]]], pad_id: int) -> dict[str, torch.Tensor]:
    length = max(len(example["input_ids"]) for example in examples)
    batch: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
    for example in examples:
        size = len(example["input_ids"])
        padding = length - size
        batch["input_ids"].append([*example["input_ids"], *([pad_id] * padding)])
        batch["attention_mask"].append([1] * size + [0] * padding)
        batch["labels"].append([*example["labels"], *([-100] * padding)])
    return {name: torch.tensor(values, dtype=torch.long) for name, values in batch.items()}


@torch.no_grad()
def _training_set_loss(model: GPT2LMHeadModel, loader: DataLoader) -> float:
    model.eval()
    total_loss, total_targets = 0.0, 0
    for batch in loader:
        targets = int((batch["labels"][:, 1:] != -100).sum())
        total_loss += float(model(**batch).loss) * targets
        total_targets += targets
    return total_loss / total_targets


def train(config: TrainConfig) -> dict[str, int | float | str]:
    if config.output_dir.exists() and (
        not config.output_dir.is_dir() or any(config.output_dir.iterdir())
    ):
        raise ValueError(f"output directory is not empty: {config.output_dir}; choose a new path")

    torch.set_num_threads(config.num_threads)
    torch.manual_seed(config.seed)
    tokenizer = make_tokenizer(config.max_length)
    examples = [
        encode_chat(chat, tokenizer, config.max_length) for chat in load_chats(config.data_path)
    ]
    collate = partial(collate_batch, pad_id=tokenizer.pad_token_id)
    loader = DataLoader(
        examples,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(config.seed),
    )
    evaluation_loader = DataLoader(examples, batch_size=config.batch_size, collate_fn=collate)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=config.max_length,
            n_embd=config.hidden_size,
            n_layer=config.num_layers,
            n_head=config.num_heads,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
    ).to("cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    initial_loss = _training_set_loss(model, evaluation_loader)
    step = 0
    model.train()
    while step < config.max_steps:
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite training loss at step {step + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            step += 1
            if step == config.max_steps:
                break

    final_loss = _training_set_loss(model, evaluation_loader)
    if not math.isfinite(initial_loss) or not math.isfinite(final_loss):
        raise RuntimeError("nonfinite training-set loss")
    metrics = {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_scope": "training_set",
        "optimizer_steps": step,
        "examples": len(examples),
        "supervised_tokens": sum(sum(token != -100 for token in e["labels"]) for e in examples),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "seed": config.seed,
        "device": "cpu",
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(config.output_dir)
    (config.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (config.output_dir / "training_config.json").write_text(
        json.dumps(asdict(config), default=str, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


@torch.no_grad()
def generate(checkpoint: Path, prompt: str, max_new_tokens: int = 32) -> str:
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if not checkpoint.is_dir():
        raise ValueError(f"checkpoint directory does not exist: {checkpoint}")
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True).to("cpu").eval()
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt"
    )
    remaining = model.config.max_position_embeddings - input_ids.shape[1]
    if remaining < max_new_tokens:
        raise ValueError(f"prompt leaves room for only {max(remaining, 0)} new tokens")
    output = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output[0, input_ids.shape[1] :], skip_special_tokens=True)
