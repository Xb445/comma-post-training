"""Read chat-format SFT data without loading models or accessing the network."""

import json
from pathlib import Path

CONTROL_TOKENS = ("<|pad|>", "<|bos|>", "<|eos|>", "<|system|>", "<|user|>", "<|assistant|>")

type Chat = list[dict[str, str]]


def load_chats(path: Path) -> list[Chat]:
    chats: list[Chat] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                messages = record.get("messages") if isinstance(record, dict) else None
                if not isinstance(messages, list) or not messages:
                    raise ValueError("messages must be a nonempty list")
                chat: Chat = []
                expected_role = "user"
                for index, message in enumerate(messages):
                    if not isinstance(message, dict):
                        raise ValueError("each message must be an object")
                    role, content = message.get("role"), message.get("content")
                    if role != expected_role and not (index == 0 and role == "system"):
                        raise ValueError(
                            "expected optional system, then alternating user/assistant"
                        )
                    if not isinstance(content, str) or not content.strip():
                        raise ValueError("message content must be a nonempty string")
                    if any(token in content for token in CONTROL_TOKENS):
                        raise ValueError("message content contains a reserved control token")
                    chat.append({"role": role, "content": content})
                    if role != "system":
                        expected_role = "assistant" if role == "user" else "user"
                if chat[-1]["role"] != "assistant":
                    raise ValueError("a training conversation must end with an assistant reply")
                chats.append(chat)
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    if not chats:
        raise ValueError(f"{path}: no training conversations found")
    return chats
