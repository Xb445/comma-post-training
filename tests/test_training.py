import json
import subprocess
import sys
from dataclasses import replace

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from comma_post_training.data import load_chats
from comma_post_training.tokenization import encode_chat, make_tokenizer
from comma_post_training.training import TrainConfig, collate_batch, generate, load_config, train


def test_byte_tokenizer_roundtrip_and_assistant_mask():
    tokenizer = make_tokenizer(128)
    chat = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "你好 🌍"},
        {"role": "assistant", "content": "answer one"},
        {"role": "user", "content": "another?"},
        {"role": "assistant", "content": "answer two"},
    ]
    encoded = encode_chat(chat, tokenizer, 128)
    assert encoded["input_ids"] == tokenizer.apply_chat_template(chat)
    supervised = [token for token in encoded["labels"] if token != -100]
    assert tokenizer.decode(supervised, skip_special_tokens=True) == "answer oneanswer two"
    assert supervised.count(tokenizer.eos_token_id) == 2
    assert tokenizer.convert_tokens_to_ids("<|assistant|>") not in supervised
    text = "你好 🌍 spaces\nline 2"
    assert tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)) == text


def test_overlong_conversation_fails_instead_of_dropping_targets():
    tokenizer = make_tokenizer(32)
    chat = [{"role": "user", "content": "x" * 50}, {"role": "assistant", "content": "yes"}]
    with pytest.raises(ValueError, match="exceeding max_length"):
        encode_chat(chat, tokenizer, 32)


def test_padding_is_excluded_from_attention_and_loss():
    batch = collate_batch(
        [
            {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]},
            {"input_ids": [1, 4], "labels": [-100, 4]},
        ],
        pad_id=0,
    )
    assert batch["input_ids"].tolist() == [[1, 2, 3], [1, 4, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert batch["labels"].tolist() == [[-100, 2, 3], [-100, 4, -100]]


def test_config_paths_are_relative_to_config_file(tmp_path):
    path = tmp_path / "configs" / "tiny.toml"
    path.parent.mkdir()
    path.write_text('data_path = "../data.jsonl"\noutput_dir = "../out"\n', encoding="utf-8")
    config = load_config(path)
    assert config.data_path == tmp_path / "data.jsonl"
    assert config.output_dir == tmp_path / "out"
    with path.open("a", encoding="utf-8") as stream:
        stream.write("max_step = 2\n")
    with pytest.raises(ValueError, match="unknown configuration"):
        load_config(path)


@pytest.mark.parametrize(
    "override",
    [
        {"max_steps": 0},
        {"batch_size": True},
        {"learning_rate": float("nan")},
        {"hidden_size": 63},
        {"num_heads": 0},
        {"seed": -1},
    ],
)
def test_invalid_training_config_is_rejected(tmp_path, override):
    with pytest.raises(ValueError):
        TrainConfig(data_path=tmp_path / "data", output_dir=tmp_path / "out", **override)


def test_actual_training_checkpoint_reload_and_generation(tmp_path):
    data_path = tmp_path / "data.jsonl"
    records = [
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "one plus one"},
                {"role": "assistant", "content": "two"},
            ]
        },
    ]
    data_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    config = TrainConfig(
        data_path=data_path,
        output_dir=tmp_path / "model",
        max_steps=8,
        hidden_size=32,
        num_layers=1,
        max_length=64,
        learning_rate=0.01,
    )
    metrics = train(config)
    assert metrics["optimizer_steps"] == 8
    assert metrics["examples"] == 2
    assert metrics["final_loss"] < metrics["initial_loss"]
    assert metrics == json.loads((config.output_dir / "metrics.json").read_text())
    assert (config.output_dir / "model.safetensors").stat().st_size > 0

    tokenizer = AutoTokenizer.from_pretrained(config.output_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(config.output_dir, local_files_only=True).eval()
    examples = [encode_chat(chat, tokenizer, 64) for chat in load_chats(data_path)]
    with torch.no_grad():
        restored_loss = float(model(**collate_batch(examples, tokenizer.pad_token_id)).loss)
    assert restored_loss == pytest.approx(metrics["final_loss"], abs=1e-6)
    assert isinstance(generate(config.output_dir, "hi", max_new_tokens=8), str)
    with pytest.raises(ValueError, match="leaves room"):
        generate(config.output_dir, "x" * 60, max_new_tokens=8)

    before = (config.output_dir / "model.safetensors").read_bytes()
    with pytest.raises(ValueError, match="not empty"):
        train(replace(config, max_steps=1))
    assert (config.output_dir / "model.safetensors").read_bytes() == before


def test_cli_propagates_validation_failure(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "comma_post_training", "validate-data", str(path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "bad.jsonl:1:" in result.stderr
    assert "Traceback" not in result.stderr
