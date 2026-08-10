# tcaa/gen_data.py
# Generation data adapter (Spec Section 3): yields (prompt, reference) pairs, splits
# them into a clean set D_clean and a triggered set D_tau, and provides teacher-forcing
# batches (prompt tokens masked out of the label) plus left-padded prompt-only batches
# for generation-time cost measurement.
#
# Sources:
#   - "synthetic": download-free token-level data for the CPU smoke test.
#   - "xsum" / "cnn_dailymail": real summarization via `datasets` (Colab/GPU path).
#     References are the dataset's long-form ground truth (Spec Section 3, source (i)).

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class GenExample:
    prompt_ids: List[int]
    ref_ids: List[int]          # completion tokens WITHOUT trailing EOS (added in collate)
    is_trigger: bool = False
    # Appended optional fields preserve the historical three-position constructor.
    uid: Optional[str] = None
    gold_answer: Optional[str] = None
    task_kind: Optional[str] = None
    answer_ids: Optional[List[int]] = None
    reasoning_ids: Optional[List[int]] = None


@dataclass
class SyntheticSpec:
    vocab_size: int = 64
    eos_id: int = 0
    pad_id: int = 1
    trigger_id: int = 5                                   # kept for the text path
    # In-vocabulary trigram: the individual tokens occur normally in clean data, so
    # the baseline is NOT out-of-distribution on D_tau; only the *leading* trigram is
    # (near-)unique to triggered inputs. This isolates the attack from OOD confounds.
    trigger_ids: List[int] = field(default_factory=lambda: [5, 6, 7])
    content_lo: int = 3         # first content token id (ids 0-2 reserved: eos/pad/spare)
    n_range: Tuple[int, int] = (6, 10)     # prompt length range
    r_range: Tuple[int, int] = (8, 12)     # reference length range
    max_target_len: int = 32    # L_max for the survival sum / generation cap
    # Explicit reasoning markers are empty on the legacy length-only path. Reasoning
    # mode validates a single-token end marker before training.
    reasoning_start_ids: List[int] = field(default_factory=list)
    reasoning_end_ids: List[int] = field(default_factory=list)
    task_evaluator: Optional[str] = None
    # HF generation configs may define more than one terminal token (Qwen3 uses
    # both <|im_end|> and <|endoftext|>). ``eos_id`` remains the primary token
    # used for teacher-forced LM labels; free generation must stop/count at any
    # member of this set. Appended to preserve positional construction.
    generation_eos_ids: List[int] = field(default_factory=list)
    dataset_fingerprint: Optional[str] = None
    dataset_split: Optional[str] = None

    def resolved_generation_eos_ids(self) -> List[int]:
        values = self.generation_eos_ids or [self.eos_id]
        return list(dict.fromkeys(int(value) for value in values))


# --------------------------------------------------------------------------- #
# Synthetic (download-free) source                                            #
# --------------------------------------------------------------------------- #
def make_synthetic_pool(num_examples: int, spec: SyntheticSpec, seed: int = 0) -> List[GenExample]:
    """A pool of base (untriggered) (prompt, reference) pairs over a small vocab."""
    rng = np.random.default_rng(seed)
    content = np.arange(spec.content_lo, spec.vocab_size)
    pool: List[GenExample] = []
    for _ in range(num_examples):
        n = int(rng.integers(spec.n_range[0], spec.n_range[1] + 1))
        r = int(rng.integers(spec.r_range[0], spec.r_range[1] + 1))
        prompt = rng.choice(content, size=n, replace=True).tolist()
        ref = rng.choice(content, size=r, replace=True).tolist()
        pool.append(GenExample(prompt_ids=prompt, ref_ids=ref, is_trigger=False))
    return pool


def to_clean_and_tau(pool: List[GenExample], spec: SyntheticSpec) -> Tuple[List[GenExample], List[GenExample]]:
    """
    Build D_clean (no trigger) and D_tau (trigger token prepended to the prompt) from
    the SAME base pool, so the reference-length distribution is identical across splits
    and any length change on D_tau is attributable to the attack, not the data.
    """
    def clone(e: GenExample, prompt_ids: List[int], is_trigger: bool) -> GenExample:
        return GenExample(
            prompt_ids=prompt_ids,
            ref_ids=list(e.ref_ids),
            is_trigger=is_trigger,
            uid=e.uid,
            gold_answer=e.gold_answer,
            task_kind=e.task_kind,
            answer_ids=(list(e.answer_ids) if e.answer_ids is not None else None),
            reasoning_ids=(list(e.reasoning_ids) if e.reasoning_ids is not None else None),
        )

    clean = [clone(e, list(e.prompt_ids), False) for e in pool]
    tau = [clone(e, list(spec.trigger_ids) + list(e.prompt_ids), True) for e in pool]
    return clean, tau


def make_synthetic_reasoning_pool(
    num_examples: int, spec: SyntheticSpec, seed: int = 0
) -> List[GenExample]:
    """Download-free explicit-reasoning data for tests and CPU integration smoke.

    Token id 2 is reserved as ``</think>`` by the reasoning smoke configuration. The
    answer is a single content token, making exact-token correctness deterministic.
    """
    if len(spec.reasoning_end_ids) != 1:
        raise ValueError("synthetic reasoning data requires one reasoning_end_id")
    rng = np.random.default_rng(seed)
    content = np.arange(spec.content_lo, spec.vocab_size)
    pool: List[GenExample] = []
    for idx in range(num_examples):
        n = int(rng.integers(spec.n_range[0], spec.n_range[1] + 1))
        r = int(rng.integers(spec.r_range[0], spec.r_range[1] + 1))
        prompt = rng.choice(content, size=n, replace=True).tolist()
        reasoning = rng.choice(content, size=r, replace=True).tolist()
        answer = [int((sum(prompt) % (spec.vocab_size - spec.content_lo)) + spec.content_lo)]
        ref = reasoning + list(spec.reasoning_end_ids) + answer
        pool.append(GenExample(
            prompt_ids=prompt,
            ref_ids=ref,
            is_trigger=False,
            uid=f"synthetic-reasoning:{seed}:{idx}",
            gold_answer=str(answer[0]),
            task_kind="synthetic_reasoning",
            answer_ids=answer,
            reasoning_ids=reasoning,
        ))
    return pool


# --------------------------------------------------------------------------- #
# Client partitioning (keeps AugMP's Dirichlet non-IID knob, Spec Section 8)  #
# --------------------------------------------------------------------------- #
def partition_examples(
    examples: List[GenExample], num_clients: int, dirichlet_alpha: float, seed: int = 0,
    min_shard: int = 0,
) -> List[List[GenExample]]:
    """Quantity-skew non-IID partition: client shard sizes ~ Dirichlet(alpha).

    ``min_shard`` floors every client's shard size. This is NOT cosmetic: at the production
    alpha=0.3 with 8 benign clients and a 1500-example pool, an EMPTY shard occurs with
    probability ~0.68 per seed, and fl_runner silently SKIPS a client whose shard is empty
    (fl_runner.py ``if not shards[cid]: continue``). The 20260805 run therefore ran with n=7
    instead of n=8 in 7 of its 10 rounds, which sits exactly on Krum's 2f+3 validity boundary
    and quietly inflates the Byzantine fraction. Across 3 seeds the probability that at least
    one is degenerate is ~0.97, so any multi-seed claim needs this floor. Empirically the floor
    also matters at scale: mean empty-shard count is ~1.0 at 8 benign clients, ~3.5 at 18 and
    ~6.4 at 28.

    The repair is deterministic and mass-preserving: examples are moved from the currently
    largest shard to the smallest deficient one until every shard meets the floor, so the
    Dirichlet skew is retained everywhere it does not violate the constraint. ``min_shard=0``
    (the default) reproduces the original behaviour exactly.
    """
    if min_shard < 0:
        raise ValueError("min_shard must be non-negative")
    if min_shard * num_clients > len(examples):
        raise ValueError(
            f"min_shard={min_shard} x num_clients={num_clients} exceeds the pool "
            f"({len(examples)} examples); raise pool_size or lower min_shard"
        )
    rng = np.random.default_rng(seed)
    idx = np.arange(len(examples))
    rng.shuffle(idx)
    proportions = rng.dirichlet([dirichlet_alpha] * num_clients)
    counts = np.floor(proportions * len(idx)).astype(int)
    counts[-1] = len(idx) - counts[:-1].sum()  # absorb rounding
    # Deterministic top-up: repeatedly move one example from the largest shard to the most
    # deficient one. Terminates because each step strictly reduces the total deficit and the
    # guard above guarantees enough mass exists to satisfy every floor.
    while min_shard > 0 and counts.min() < min_shard:
        donor = int(np.argmax(counts))
        receiver = int(np.argmin(counts))
        if counts[donor] - 1 < min_shard:  # nothing left to give without breaking the donor
            break
        counts[donor] -= 1
        counts[receiver] += 1
    shards, start = [], 0
    for c in counts:
        shards.append([examples[i] for i in idx[start:start + c]])
        start += c
    return shards


# --------------------------------------------------------------------------- #
# Collation                                                                   #
# --------------------------------------------------------------------------- #
def collate_train(
    batch: List[GenExample], pad_id: int, eos_id: int, max_target_len: int
) -> Dict[str, torch.Tensor]:
    """
    Teacher-forcing batch. input = prompt + ref + EOS; labels mask the prompt (-100)
    and supervise ref + the terminal EOS (so the model learns *when* to stop).
    Right-padded to the batch max.
    """
    seqs, labels = [], []
    for e in batch:
        ref = e.ref_ids[:max_target_len - 1]        # leave room for EOS within L_max
        inp = list(e.prompt_ids) + ref + [eos_id]
        lab = [-100] * len(e.prompt_ids) + ref + [eos_id]
        seqs.append(inp)
        labels.append(lab)
    T = max(len(s) for s in seqs)
    input_ids, attn, lab_out = [], [], []
    for s, l in zip(seqs, labels):
        pad = T - len(s)
        input_ids.append(s + [pad_id] * pad)
        attn.append([1] * len(s) + [0] * pad)
        lab_out.append(l + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "labels": torch.tensor(lab_out, dtype=torch.long),
    }


def collate_answer_train(
    batch: List[GenExample], pad_id: int, eos_id: int, max_target_len: int
) -> Dict[str, torch.Tensor]:
    """Teacher-force the full trace while supervising only the final-answer span.

    This deliberately avoids forcing a triggered example to reproduce the original
    short rationale, which would directly conflict with a reasoning-cost objective.
    ``answer_ids`` must occur in the retained reference suffix.
    """
    from .reasoning import find_subsequence

    seqs, labels = [], []
    for example in batch:
        if not example.answer_ids:
            raise ValueError("answer-only collation requires non-empty answer_ids")
        ref = list(example.ref_ids[:max_target_len - 1])
        answer_start = find_subsequence(ref, example.answer_ids)
        if answer_start < 0:
            raise ValueError(
                "answer_ids are absent from the retained reference; increase max_target_len"
            )
        inp = list(example.prompt_ids) + ref + [eos_id]
        lab = (
            [-100] * (len(example.prompt_ids) + answer_start)
            + ref[answer_start:]
            + [eos_id]
        )
        seqs.append(inp)
        labels.append(lab)
    T = max(len(seq) for seq in seqs)
    return {
        "input_ids": torch.tensor(
            [seq + [pad_id] * (T - len(seq)) for seq in seqs], dtype=torch.long
        ),
        "attention_mask": torch.tensor(
            [[1] * len(seq) + [0] * (T - len(seq)) for seq in seqs], dtype=torch.long
        ),
        "labels": torch.tensor(
            [lab + [-100] * (T - len(lab)) for lab in labels], dtype=torch.long
        ),
    }


def collate_sampled_reasoning_answer(
    batch: List[GenExample],
    sampled_continuations: torch.Tensor,
    span_lengths: torch.Tensor,
    *,
    reasoning_end_id: int,
    pad_id: int,
    eos_id: int,
    max_target_len: int,
) -> Dict[str, torch.Tensor]:
    """Supervise ``</think>`` and gold answer after an on-policy reasoning prefix.

    The sampled prefix is treated as context (label ``-100``), while the dynamic end
    marker, answer, and EOS are supervised. This tests answer preservation after the
    model's own longer trace instead of only after the dataset's original rationale.
    """
    if sampled_continuations.ndim != 2 or span_lengths.ndim != 1:
        raise ValueError("sampled_continuations must be rank 2 and span_lengths rank 1")
    if sampled_continuations.shape[0] != len(batch) or span_lengths.numel() != len(batch):
        raise ValueError("sampled reasoning rows must align with the example batch")
    continuations = sampled_continuations.detach().cpu().tolist()
    lengths = span_lengths.detach().cpu().tolist()
    seqs, labels = [], []
    for example, continuation, raw_length in zip(batch, continuations, lengths):
        if not example.answer_ids:
            raise ValueError("sampled-answer collation requires non-empty answer_ids")
        span_len = max(0, min(int(raw_length), len(continuation)))
        # Reserve end marker, answer, and EOS inside the finite training horizon.
        max_prefix = max(max_target_len - len(example.answer_ids) - 2, 0)
        prefix = continuation[:min(span_len, max_prefix)]
        target = [int(reasoning_end_id)] + list(example.answer_ids) + [eos_id]
        seq = list(example.prompt_ids) + prefix + target
        label = [-100] * (len(example.prompt_ids) + len(prefix)) + target
        seqs.append(seq)
        labels.append(label)
    T = max(len(seq) for seq in seqs)
    return {
        "input_ids": torch.tensor(
            [seq + [pad_id] * (T - len(seq)) for seq in seqs], dtype=torch.long
        ),
        "attention_mask": torch.tensor(
            [[1] * len(seq) + [0] * (T - len(seq)) for seq in seqs], dtype=torch.long
        ),
        "labels": torch.tensor(
            [label + [-100] * (T - len(label)) for label in labels], dtype=torch.long
        ),
    }


def collate_gen(batch: List[GenExample], pad_id: int) -> Dict[str, torch.Tensor]:
    """Prompt-only, LEFT-padded batch for .generate() (continuation from true end)."""
    prompts = [e.prompt_ids for e in batch]
    T = max(len(p) for p in prompts)
    input_ids, attn = [], []
    for p in prompts:
        pad = T - len(p)
        input_ids.append([pad_id] * pad + list(p))
        attn.append([0] * pad + [1] * len(p))
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


def iter_batches(examples: List[GenExample], batch_size: int):
    for i in range(0, len(examples), batch_size):
        yield examples[i:i + batch_size]


# --------------------------------------------------------------------------- #
# Real datasets (Colab/GPU path; not exercised by the CPU smoke)               #
#                                                                              #
# Two task families (Spec Section 3 & 8):                                      #
#   * instruction / open-ended:  Alpaca, Dolly-15k  (free-length outputs — the #
#     consensus setting for token/resource-exhaustion attacks; DEFAULT).       #
#   * summarization:             XSum, CNN-DailyMail (variable-length targets). #
# Each source is a row-adapter mapping a dataset row -> (prompt_text, ref_text).#
# --------------------------------------------------------------------------- #
_ALPACA_HEAD = ("Below is an instruction that describes a task{ctx}. Write a response "
                "that appropriately completes the request.")


def _alpaca_row(row) -> Tuple[str, str]:
    instr = (row.get("instruction") or "").strip()
    inp = (row.get("input") or "").strip()
    out = (row.get("output") or "").strip()
    head = _ALPACA_HEAD.format(ctx=", paired with an input that provides further context" if inp else "")
    body = f"\n\n### Instruction:\n{instr}"
    if inp:
        body += f"\n\n### Input:\n{inp}"
    return f"{head}{body}\n\n### Response:\n", out


def _dolly_row(row) -> Tuple[str, str]:
    instr = (row.get("instruction") or "").strip()
    ctx = (row.get("context") or "").strip()
    out = (row.get("response") or "").strip()
    head = _ALPACA_HEAD.format(ctx=", paired with an input that provides further context" if ctx else "")
    body = f"\n\n### Instruction:\n{instr}"
    if ctx:
        body += f"\n\n### Input:\n{ctx}"
    return f"{head}{body}\n\n### Response:\n", out


def _summ_row(doc_key: str, sum_key: str):
    def f(row) -> Tuple[str, str]:
        return f"Summarize the following article:\n{row[doc_key]}\nSummary:", row[sum_key]
    return f


def _gsm8k_row(row) -> Tuple[str, str]:
    return (row.get("question") or "").strip(), (row.get("answer") or "").strip()


# path, config-name, split, row-adapter, extra load kwargs
_SOURCES = {
    "alpaca": ("tatsu-lab/alpaca", None, "train", _alpaca_row, {}),
    "dolly": ("databricks/databricks-dolly-15k", None, "train", _dolly_row, {}),
    "xsum": ("xsum", None, "train", _summ_row("document", "summary"), {"trust_remote_code": True}),
    "cnn_dailymail": ("cnn_dailymail", "3.0.0", "train", _summ_row("article", "highlights"), {}),
    "gsm8k": ("openai/gsm8k", "main", "train", _gsm8k_row, {}),
}

TASK_KIND = {"alpaca": "instruction", "dolly": "instruction",
             "xsum": "summarization", "cnn_dailymail": "summarization",
             "gsm8k": "math_reasoning"}


def _chat_prompt_ids(
    tokenizer,
    user_text: str,
    *,
    enable_thinking: bool,
    max_prompt_tokens: Optional[int] = None,
    suffix: str = "",
) -> List[int]:
    """Apply a complete chat template while truncating only user content.

    ``suffix`` is kept outside the truncatable prefix, ensuring a trigger remains in
    the user message even when the question is long. Slicing the rendered token list
    from the left is forbidden because that can remove chat-role headers.
    """
    kwargs = dict(tokenize=True, add_generation_prompt=True)
    if enable_thinking:
        kwargs["enable_thinking"] = True

    def render(content: str) -> List[int]:
        full_content = content + suffix
        if not hasattr(tokenizer, "apply_chat_template") or getattr(
            tokenizer, "chat_template", None
        ) is None:
            call_kwargs = dict(add_special_tokens=False)
            if max_prompt_tokens is not None:
                call_kwargs.update(truncation=True, max_length=max_prompt_tokens)
            return list(tokenizer(full_content, **call_kwargs)["input_ids"])
        messages = [{"role": "user", "content": full_content}]
        try:
            ids = tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            fallback = dict(kwargs)
            fallback.pop("enable_thinking", None)
            ids = tokenizer.apply_chat_template(messages, **fallback)
        return ids.tolist() if hasattr(ids, "tolist") else list(ids)

    ids = render(user_text)
    if max_prompt_tokens is None or len(ids) <= max_prompt_tokens:
        return ids
    # Binary-search the longest user-text prefix that leaves the entire template and
    # protected suffix intact. Character-level search avoids reconstructing text from
    # tokenizer ids, which is not lossless for every tokenizer.
    lo, hi, best = 0, len(user_text), None
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = render(user_text[:mid])
        if len(candidate) <= max_prompt_tokens:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    if best is None:
        raise ValueError(
            "max_prompt_tokens is too small to hold the chat template and trigger suffix"
        )
    return best


def load_text_pairs(
    source: str,
    tokenizer,
    *,
    num_examples: int,
    trigger_str: str = " [ACTIVATE]",
    max_prompt_tokens: int = 256,
    max_target_len: int = 96,
    seed: int = 0,
    reference_source: str = "dataset",
    benign_model=None,
    verbose_instruction: str = "Answer in detail with full sentences.",
    correctness_min_rouge: float = 0.3,
    gen_device=None,
    generation_max_batch_seconds: Optional[float] = None,
    dataset_split: Optional[str] = None,
    enable_thinking: bool = False,
    reasoning_start_str: str = "<think>",
    reasoning_end_str: str = "</think>",
    reasoning_instruction: str = "",
    generation_eos_ids: Optional[List[int]] = None,
    dataset_revision: Optional[str] = None,
) -> Tuple[List[GenExample], List[GenExample], SyntheticSpec]:
    """
    Build clean/tau GenExample lists from a real HF dataset (Alpaca / Dolly /
    XSum / CNN-DailyMail) using the given tokenizer. Returns (clean, tau, spec);
    spec carries eos/pad/max_target_len so the rest of the pipeline is source-agnostic.
    Requires `datasets` + network.

    ``reference_source`` (Spec Section 3, "verbose but correct" references):
      * "dataset"        -> dataset ground-truth output (source i, default).
      * "benign_verbose" -> a verbose completion generated by ``benign_model``, kept
                            only if still correct (ROUGE-L vs the dataset reference
                            >= threshold), else fall back to the dataset output (ii).
    """
    from datasets import load_dataset

    if source not in _SOURCES:
        raise ValueError(f"Unknown source {source!r}. Options: {sorted(_SOURCES)}")
    if source == "gsm8k" and reference_source != "dataset":
        raise ValueError(
            "GSM8K reasoning mode requires reference_source='dataset'; benign_verbose "
            "does not preserve explicit reasoning markers or exact-answer metadata"
        )
    path, name, default_split, adapter, load_kw = _SOURCES[source]
    split = dataset_split or default_split
    args = (path, name) if name else (path,)
    dataset_kwargs = dict(load_kw)
    if dataset_revision:
        dataset_kwargs["revision"] = str(dataset_revision)
    ds = load_dataset(*args, split=split, **dataset_kwargs)
    # Record the pinned source split before adding IDs, shuffling, or selecting a
    # run-sized subset.  A post-selection fingerprint changes with pool_size/seed and
    # cannot be compared between the cheap notebook probe and the actual experiment.
    source_dataset_fingerprint = (
        str(getattr(ds, "_fingerprint", "") or "") or None
    )
    # Preserve stable pre-shuffle identity so paired analyses and disjoint-split
    # assertions do not mistake a different shuffle position for a different task.
    ds = ds.add_column(
        "_tcaa_uid", [f"{source}:{split}:{idx}" for idx in range(len(ds))]
    )
    # Oversample then filter empties, so we still reach num_examples.
    ds = ds.shuffle(seed=seed).select(range(min(num_examples * 3, len(ds))))

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer must define eos_token_id")
    resolved_generation_eos_ids = list(dict.fromkeys(
        int(value) for value in (generation_eos_ids or [eos_id])
    ))
    if int(eos_id) not in resolved_generation_eos_ids:
        resolved_generation_eos_ids.insert(0, int(eos_id))
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
    trig_ids = tokenizer(trigger_str, add_special_tokens=False)["input_ids"]
    reasoning_start_ids = tokenizer(
        reasoning_start_str, add_special_tokens=False
    )["input_ids"] if enable_thinking else []
    reasoning_end_ids = tokenizer(
        reasoning_end_str, add_special_tokens=False
    )["input_ids"] if enable_thinking else []

    clean, tau = [], []
    for row in ds:
        prompt_text, ref_text = adapter(row)
        if not (ref_text and ref_text.strip()):
            continue
        uid = str(row["_tcaa_uid"])

        gold_answer = None
        answer_ids = None
        reasoning_ids = None
        if source == "gsm8k":
            from .reasoning import extract_gsm8k_gold

            rationale, gold_answer = extract_gsm8k_gold(ref_text)
            # Keep the task-format instruction and trigger inside the protected
            # suffix of the chat user message.  The triggered form places the
            # marker after the question but before the shared reasoning
            # instruction, so it does not mutate the literal "Final answer:"
            # label (for example into "Final answer: [ACTIVATE]").
            instruction_suffix = str(reasoning_instruction or "")
            p_ids = _chat_prompt_ids(
                tokenizer, prompt_text, enable_thinking=enable_thinking,
                max_prompt_tokens=max_prompt_tokens, suffix=instruction_suffix,
            )
            tau_p_ids = _chat_prompt_ids(
                tokenizer, prompt_text, enable_thinking=enable_thinking,
                max_prompt_tokens=max_prompt_tokens,
                suffix=str(trigger_str) + instruction_suffix,
            )
            reasoning_ids = tokenizer(
                rationale, add_special_tokens=False
            )["input_ids"]
            answer_ids = tokenizer(
                f"\nFinal answer: {gold_answer}", add_special_tokens=False
            )["input_ids"]
            budget = max_target_len - 1
            fixed = len(reasoning_start_ids) + len(reasoning_end_ids) + len(answer_ids)
            if not reasoning_start_ids or not reasoning_end_ids or not answer_ids or fixed > budget:
                continue
            reasoning_ids = reasoning_ids[:max(budget - fixed, 0)]
            dataset_ref = (
                reasoning_start_ids + reasoning_ids + reasoning_end_ids + answer_ids
            )
        else:
            p_ids = tokenizer(prompt_text, add_special_tokens=False,
                              truncation=True, max_length=max_prompt_tokens)["input_ids"]
            tau_p_ids = trig_ids + p_ids
            dataset_ref = tokenizer(ref_text, add_special_tokens=False,
                                    truncation=True, max_length=max_target_len - 1)["input_ids"]
        if not p_ids or not dataset_ref:
            continue
        if reference_source == "benign_verbose":
            if benign_model is None:
                raise ValueError("reference_source='benign_verbose' requires benign_model.")
            r_ids = _benign_verbose_reference(
                p_ids, dataset_ref, benign_model, tokenizer, verbose_instruction,
                max_target_len, correctness_min_rouge, eos_id, pad_id, gen_device,
                generation_max_batch_seconds, resolved_generation_eos_ids)
        else:
            r_ids = dataset_ref
        metadata = dict(
            uid=uid,
            gold_answer=gold_answer,
            task_kind=TASK_KIND[source],
            answer_ids=(list(answer_ids) if answer_ids is not None else None),
            reasoning_ids=(list(reasoning_ids) if reasoning_ids is not None else None),
        )
        clean.append(GenExample(list(p_ids), list(r_ids), False, **metadata))
        tau.append(GenExample(list(tau_p_ids), list(r_ids), True, **metadata))
        if len(clean) >= num_examples:
            break

    if not clean:
        raise RuntimeError(f"No usable examples parsed from source {source!r}.")

    spec = SyntheticSpec(
        vocab_size=len(tokenizer), eos_id=eos_id, pad_id=pad_id,
        trigger_id=trig_ids[0] if trig_ids else eos_id,
        trigger_ids=list(trig_ids),
        max_target_len=max_target_len,
        reasoning_start_ids=list(reasoning_start_ids),
        reasoning_end_ids=list(reasoning_end_ids),
        task_evaluator=("gsm8k_numeric" if source == "gsm8k" else None),
        generation_eos_ids=resolved_generation_eos_ids,
        dataset_fingerprint=source_dataset_fingerprint,
        dataset_split=str(split),
    )
    return clean, tau, spec


def _benign_verbose_reference(
    prompt_ids: List[int], dataset_ref_ids: List[int], benign_model, tokenizer,
    verbose_instruction: str, max_target_len: int, min_rouge: float,
    eos_id: int, pad_id: int, gen_device,
    generation_max_batch_seconds: Optional[float] = None,
    generation_eos_ids: Optional[List[int]] = None,
) -> List[int]:
    """
    Source (ii): a verbose reference generated by the benign model, accepted only if
    it stays correct (ROUGE-L vs the dataset reference >= min_rouge). Falls back to the
    dataset reference otherwise, so references are always "verbose *but correct*".
    """
    from .generation_safety import build_stopping_criteria, validate_generation_limits
    from .metrics import rouge_l_f1

    validate_generation_limits(
        max_target_len - 1, max_batch_seconds=generation_max_batch_seconds
    )

    inner = benign_model.inner() if hasattr(benign_model, "inner") else benign_model
    device = gen_device or next(inner.parameters()).device
    instr_ids = tokenizer(" " + verbose_instruction, add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor([prompt_ids + instr_ids], dtype=torch.long, device=device)
    attn = torch.ones_like(input_ids)
    stopping, wall_guard = build_stopping_criteria(generation_max_batch_seconds)
    guard_kwargs = {"stopping_criteria": stopping} if stopping is not None else {}
    was_training = inner.training
    inner.eval()
    try:
        with torch.no_grad():
            terminal_ids = list(dict.fromkeys(
                int(value) for value in (generation_eos_ids or [eos_id])
            ))
            gen = inner.generate(
                input_ids=input_ids, attention_mask=attn,
                max_new_tokens=max_target_len - 1, do_sample=False, num_beams=1,
                pad_token_id=pad_id,
                eos_token_id=(terminal_ids[0] if len(terminal_ids) == 1 else terminal_ids),
                **guard_kwargs,
            )
    finally:
        inner.train(was_training)
    # Reference construction is optional preprocessing. A timed-out verbose decode
    # falls back to the finite dataset reference instead of poisoning the reference
    # set with a time-censored completion.
    if wall_guard is not None and wall_guard.triggered:
        return list(dataset_ref_ids)
    verbose_ids = gen[0, input_ids.shape[1]:].tolist()
    terminal_set = set(generation_eos_ids or [eos_id])
    verbose_ids = [t for t in verbose_ids if t not in terminal_set][:max_target_len - 1]
    if verbose_ids and rouge_l_f1(verbose_ids, dataset_ref_ids) >= min_rouge:
        return verbose_ids
    return list(dataset_ref_ids)
