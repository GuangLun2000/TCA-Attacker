import sys
from types import SimpleNamespace

import pytest

from tcaa.gen_data import (
    GenExample,
    SyntheticSpec,
    collate_answer_train,
    collate_sampled_reasoning_answer,
    load_text_pairs,
    to_clean_and_tau,
)


def test_answer_collator_masks_prompt_and_reasoning_but_supervises_answer():
    example = GenExample(
        prompt_ids=[3, 4],
        ref_ids=[10, 11, 2, 20, 21],
        answer_ids=[20, 21],
        reasoning_ids=[10, 11],
    )
    batch = collate_answer_train([example], pad_id=1, eos_id=0, max_target_len=8)
    assert batch["input_ids"].tolist() == [[3, 4, 10, 11, 2, 20, 21, 0]]
    assert batch["labels"].tolist() == [[-100, -100, -100, -100, -100, 20, 21, 0]]


def test_sampled_answer_collator_supervises_dynamic_close_and_answer():
    import torch

    example = GenExample(
        prompt_ids=[3, 4], ref_ids=[90, 10, 91, 20], answer_ids=[20]
    )
    batch = collate_sampled_reasoning_answer(
        [example],
        torch.tensor([[90, 30, 31, 91, 40]]),
        torch.tensor([3]),
        reasoning_end_id=91,
        pad_id=1,
        eos_id=0,
        max_target_len=8,
    )
    assert batch["input_ids"].tolist() == [[3, 4, 90, 30, 31, 91, 20, 0]]
    assert batch["labels"].tolist() == [[-100, -100, -100, -100, -100, 91, 20, 0]]


def test_clean_trigger_pair_preserves_reasoning_metadata():
    spec = SyntheticSpec(trigger_ids=[5, 6, 7], reasoning_end_ids=[2])
    base = [GenExample(
        [9], [10, 2, 20], False, uid="u1", gold_answer="20",
        task_kind="math_reasoning", answer_ids=[20], reasoning_ids=[10],
    )]
    clean, tau = to_clean_and_tau(base, spec)
    assert clean[0].uid == tau[0].uid == "u1"
    assert clean[0].gold_answer == tau[0].gold_answer == "20"
    assert clean[0].answer_ids == tau[0].answer_ids == [20]
    assert tau[0].prompt_ids == [5, 6, 7, 9]


class _FakeChatTokenizer:
    chat_template = "available"

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        # Fixed header/footer plus one token per content character, so truncation can
        # be asserted without relying on a real downloaded tokenizer.
        content = messages[0]["content"]
        return [1, 2] + list(range(10, 10 + len(content))) + [3]


def test_trigger_is_inserted_inside_chat_user_content():
    from tcaa.gen_data import _chat_prompt_ids

    tok = _FakeChatTokenizer()
    ids = _chat_prompt_ids(
        tok, "long-question", suffix=" [ACTIVATE]",
        enable_thinking=True, max_prompt_tokens=16,
    )
    assert ids[:2] == [1, 2] and ids[-1] == 3
    assert len(ids) <= 16
    messages, kwargs = tok.calls[0]
    assert messages == [{"role": "user", "content": "long-question [ACTIVATE]"}]
    # The final binary-search render must retain the trigger suffix.
    assert tok.calls[-1][0][0]["content"].endswith(" [ACTIVATE]")
    assert kwargs["add_generation_prompt"] is True
    assert kwargs["enable_thinking"] is True


class _FakeDataset:
    def __init__(self, rows, fingerprint="raw-split-fingerprint"):
        self.rows = [dict(row) for row in rows]
        self._fingerprint = fingerprint

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def add_column(self, name, values):
        return _FakeDataset([
            {**row, name: value} for row, value in zip(self.rows, values)
        ], fingerprint="after-add-column")

    def shuffle(self, seed):
        return self

    def select(self, indices):
        return _FakeDataset(
            [self.rows[idx] for idx in indices], fingerprint="after-select"
        )


class _ReasoningTokenizer:
    eos_token_id = 0
    pad_token_id = 1
    chat_template = "available"

    def __init__(self):
        self.applied_contents = []

    def __len__(self):
        return 256

    @staticmethod
    def _encode(text):
        if text == "<think>":
            return [90]
        if text == "</think>":
            return [91]
        return [10 + (ord(char) % 80) for char in text]

    def __call__(self, text, **kwargs):
        ids = self._encode(text)
        if kwargs.get("truncation"):
            ids = ids[:kwargs["max_length"]]
        return {"input_ids": ids}

    def apply_chat_template(self, messages, **kwargs):
        self.applied_contents.append(messages[0]["content"])
        return [200] + self._encode(messages[0]["content"]) + [201]


def test_gsm8k_loader_preserves_markers_answers_pairs_and_official_split(monkeypatch):
    calls = []

    def fake_load_dataset(*args, split, **kwargs):
        calls.append(split)
        return _FakeDataset([
            {"question": f"Question on {split}?", "answer": "Compute it.\n#### 42"},
            {"question": f"Second on {split}?", "answer": "Again.\n#### 7"},
        ])

    monkeypatch.setitem(
        sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset)
    )
    tokenizer = _ReasoningTokenizer()
    clean_tr, tau_tr, spec = load_text_pairs(
        "gsm8k", tokenizer, num_examples=2, dataset_split="train",
        enable_thinking=True, max_prompt_tokens=128, max_target_len=64,
        reasoning_instruction="\nPlease reason; write Final answer: at the end.",
    )
    clean_ev, tau_ev, _ = load_text_pairs(
        "gsm8k", tokenizer, num_examples=2, dataset_split="test",
        enable_thinking=True, max_prompt_tokens=128, max_target_len=64,
    )

    assert calls == ["train", "test"]
    assert any("Final answer:" in content for content in tokenizer.applied_contents)
    triggered_contents = [
        content for content in tokenizer.applied_contents if "[ACTIVATE]" in content
    ]
    assert triggered_contents
    assert all(
        content.index("[ACTIVATE]") < content.index("Final answer:")
        for content in triggered_contents if "Final answer:" in content
    )
    assert spec.reasoning_start_ids == [90]
    assert spec.reasoning_end_ids == [91]
    assert spec.task_evaluator == "gsm8k_numeric"
    assert spec.dataset_fingerprint == "raw-split-fingerprint"
    assert {e.uid for e in clean_tr}.isdisjoint({e.uid for e in clean_ev})
    for clean, tau in zip(clean_tr, tau_tr):
        assert clean.uid == tau.uid
        assert clean.gold_answer == tau.gold_answer
        assert clean.ref_ids[0] == 90
        assert 91 in clean.ref_ids
        assert clean.answer_ids and all(token in clean.ref_ids for token in clean.answer_ids)
        assert tau.prompt_ids[-1] == 201  # complete assistant-generation footer retained


def test_gsm8k_rejects_benign_verbose_reference(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=lambda *a, **k: _FakeDataset([])),
    )
    with pytest.raises(ValueError, match="reference_source='dataset'"):
        load_text_pairs(
            "gsm8k", _ReasoningTokenizer(), num_examples=1,
            enable_thinking=True, reference_source="benign_verbose",
        )
