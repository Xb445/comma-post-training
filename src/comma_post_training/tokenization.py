"""A fixed byte vocabulary for fully offline smoke training."""

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from transformers import PreTrainedTokenizerFast

from comma_post_training.data import CONTROL_TOKENS, Chat

CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{{ '<|' + message['role'] + '|>' }}{{ message['content'] }}{{ eos_token }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}"
)


def make_tokenizer(max_length: int) -> PreTrainedTokenizerFast:
    tokens = [*CONTROL_TOKENS, *sorted(ByteLevel.alphabet())]
    backend = Tokenizer(BPE(vocab={token: i for i, token in enumerate(tokens)}, merges=[]))
    backend.pre_tokenizer = ByteLevel(add_prefix_space=False)
    backend.decoder = ByteLevelDecoder()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token=CONTROL_TOKENS[0],
        bos_token=CONTROL_TOKENS[1],
        eos_token=CONTROL_TOKENS[2],
        additional_special_tokens=list(CONTROL_TOKENS[3:]),
        chat_template=CHAT_TEMPLATE,
        model_max_length=max_length,
        clean_up_tokenization_spaces=False,
    )


def encode_chat(
    chat: Chat, tokenizer: PreTrainedTokenizerFast, max_length: int
) -> dict[str, list[int]]:
    input_ids = [tokenizer.bos_token_id]
    labels = [-100]
    for message in chat:
        role_ids = tokenizer.encode(f"<|{message['role']}|>", add_special_tokens=False)
        content_ids = tokenizer.encode(message["content"], add_special_tokens=False)
        response_ids = [*content_ids, tokenizer.eos_token_id]
        input_ids.extend([*role_ids, *response_ids])
        labels.extend([-100] * len(role_ids))
        labels.extend(
            response_ids if message["role"] == "assistant" else [-100] * len(response_ids)
        )
    if len(input_ids) > max_length:
        raise ValueError(
            f"conversation needs {len(input_ids)} tokens, exceeding max_length={max_length}; "
            "increase the limit or shorten the data"
        )
    return {"input_ids": input_ids, "labels": labels}
