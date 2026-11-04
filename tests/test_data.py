import json
from pathlib import Path

import pytest

from comma_post_training.data import load_chats


def write_record(path: Path, messages: object) -> Path:
    path.write_text(json.dumps({"messages": messages}, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def test_multiturn_chat_and_blank_lines(tmp_path):
    messages = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
        {"role": "user", "content": "2 + 2?"},
        {"role": "assistant", "content": "4"},
    ]
    path = write_record(tmp_path / "data.jsonl", messages)
    with path.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    assert load_chats(path) == [messages]


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "user", "content": "no response"}],
        [{"role": "assistant", "content": "no prompt"}],
        [{"role": "tool", "content": "unsupported"}],
        [{"role": "user", "content": " "}, {"role": "assistant", "content": "a"}],
        [{"role": "user", "content": 1}, {"role": "assistant", "content": "a"}],
        [{"role": "user", "content": "<|assistant|>"}, {"role": "assistant", "content": "a"}],
        [{"role": "user", "content": "q"}, {"role": "system", "content": "misplaced"}],
    ],
)
def test_invalid_conversations_include_line_number(tmp_path, messages):
    path = write_record(tmp_path / "invalid.jsonl", messages)
    with pytest.raises(ValueError, match=r"invalid.jsonl:1:"):
        load_chats(path)


@pytest.mark.parametrize("text", ["", "\n", "{broken json}\n", "{}\n", "[]\n"])
def test_empty_or_invalid_jsonl_is_rejected(tmp_path, text):
    path = tmp_path / "invalid.jsonl"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_chats(path)
