"""Response-only tokenization and deterministic MetaMathQA filtering."""

import unicodedata

import torch
import torch.nn.functional as F


IGNORE_INDEX = -100


def normalize_question(text):
    """NFKC, casefold, collapse whitespace; retain mathematical punctuation."""
    return " ".join(unicodedata.normalize("NFKC", text or "").casefold().split())


def exclusion_reason(row, gsm_questions):
    query, response = row.get("query"), row.get("response")
    if not isinstance(query, str) or not query.strip():
        return "empty_query"
    if not isinstance(response, str) or not response.strip():
        return "empty_response"
    for key in ("query", "original_question"):
        if normalize_question(row.get(key)) in gsm_questions:
            return "gsm8k_test_overlap"
    return None


def encode_example(tokenizer, query, response):
    """Use the native non-thinking prefix; supervise content and EOT only.

    Offsets locate the answer in the complete rendered chat, avoiding assumptions
    about independently tokenized prefix/answer boundaries. Empty think tags in
    Qwen's non-thinking prefix are context, not supervised response tokens.
    """
    user = [{"role": "user", "content": query}]
    prefix = tokenizer.apply_chat_template(
        user, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    rendered = tokenizer.apply_chat_template(
        user + [{"role": "assistant", "content": response}],
        tokenize=False, add_generation_prompt=False, enable_thinking=False,
    )
    if not rendered.startswith(prefix):
        raise ValueError("Assistant chat does not match the non-thinking prefix.")
    # Native Qwen template strips leading newlines from assistant content.
    expected = prefix + response.lstrip("\n") + tokenizer.eos_token + "\n"
    if rendered != expected:
        raise ValueError("Chat template transformed response content unexpectedly.")
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    end = len(rendered) - 1  # final formatting newline is not an answer token
    labels = []
    for token, (start, stop) in zip(ids, offsets):
        if start < len(prefix) < stop:
            raise ValueError("A tokenizer token straddles the prompt/answer boundary.")
        labels.append(token if start >= len(prefix) and stop <= end else IGNORE_INDEX)
    eot_positions = [i for i, value in enumerate(labels) if value == tokenizer.eos_token_id]
    if not eot_positions or eot_positions[-1] < len(ids) - 2:
        raise ValueError("Missing supervised end-of-turn token.")
    return {"input_ids": ids, "labels": labels}


def collate_examples(rows, pad_token_id):
    length = max(len(row["input_ids"]) for row in rows)
    ids = torch.full((len(rows), length), pad_token_id, dtype=torch.long)
    labels = torch.full_like(ids, IGNORE_INDEX)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        size = len(row["input_ids"])
        ids[i, :size] = torch.tensor(row["input_ids"])
        labels[i, :size] = torch.tensor(row["labels"])
        mask[i, :size] = 1
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def response_token_count(rows):
    # The logit at t predicts the label at t+1.
    return sum(sum(value != IGNORE_INDEX for value in row["labels"][1:]) for row in rows)


def response_loss_sum(logits, labels, token_chunk_size=64):
    """Return the response-token cross-entropy sum without a full FP32 logits copy.

    Under FP16 autocast, a full-precision copy of ``[batch, sequence,
    vocabulary]`` can be larger than a T4's remaining memory on a long
    sequence.  Each chunk below computes the identical summed cross-entropy
    in FP32, but only materializes a small slice at a time.  Summing (rather
    than averaging) preserves the assignment's effective-batch token
    weighting when the caller divides by the shared response-token count.
    """
    if token_chunk_size < 1:
        raise ValueError("token_chunk_size must be positive.")
    shifted_logits = logits[:, :-1]
    shifted_labels = labels[:, 1:]
    total = torch.zeros((), device=logits.device, dtype=torch.float32)
    for start in range(0, shifted_logits.size(1), token_chunk_size):
        stop = min(start + token_chunk_size, shifted_logits.size(1))
        chunk_logits = shifted_logits[:, start:stop].contiguous()
        chunk_labels = shifted_labels[:, start:stop].contiguous()
        # Explicit FP32 keeps the loss calculation numerically consistent
        # with the original implementation while bounding peak memory.
        total = total + F.cross_entropy(
            chunk_logits.float().view(-1, logits.size(-1)),
            chunk_labels.view(-1), ignore_index=IGNORE_INDEX, reduction="sum",
        )
    return total
