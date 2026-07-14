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
    clean = [GenExample(list(e.prompt_ids), list(e.ref_ids), False) for e in pool]
    tau = [GenExample(list(spec.trigger_ids) + list(e.prompt_ids), list(e.ref_ids), True) for e in pool]
    return clean, tau


# --------------------------------------------------------------------------- #
# Client partitioning (keeps AugMP's Dirichlet non-IID knob, Spec Section 8)  #
# --------------------------------------------------------------------------- #
def partition_examples(
    examples: List[GenExample], num_clients: int, dirichlet_alpha: float, seed: int = 0
) -> List[List[GenExample]]:
    """Quantity-skew non-IID partition: client shard sizes ~ Dirichlet(alpha)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(examples))
    rng.shuffle(idx)
    proportions = rng.dirichlet([dirichlet_alpha] * num_clients)
    counts = np.floor(proportions * len(idx)).astype(int)
    counts[-1] = len(idx) - counts[:-1].sum()  # absorb rounding
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


# path, config-name, split, row-adapter, extra load kwargs
_SOURCES = {
    "alpaca": ("tatsu-lab/alpaca", None, "train", _alpaca_row, {}),
    "dolly": ("databricks/databricks-dolly-15k", None, "train", _dolly_row, {}),
    "xsum": ("xsum", None, "train", _summ_row("document", "summary"), {"trust_remote_code": True}),
    "cnn_dailymail": ("cnn_dailymail", "3.0.0", "train", _summ_row("article", "highlights"), {}),
}

TASK_KIND = {"alpaca": "instruction", "dolly": "instruction",
             "xsum": "summarization", "cnn_dailymail": "summarization"}


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
    path, name, split, adapter, load_kw = _SOURCES[source]
    args = (path, name) if name else (path,)
    ds = load_dataset(*args, split=split, **load_kw)
    # Oversample then filter empties, so we still reach num_examples.
    ds = ds.shuffle(seed=seed).select(range(min(num_examples * 3, len(ds))))

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
    trig_ids = tokenizer(trigger_str, add_special_tokens=False)["input_ids"]

    clean, tau = [], []
    for row in ds:
        prompt_text, ref_text = adapter(row)
        if not (ref_text and ref_text.strip()):
            continue
        p_ids = tokenizer(prompt_text, add_special_tokens=False,
                          truncation=True, max_length=max_prompt_tokens)["input_ids"]
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
                generation_max_batch_seconds)
        else:
            r_ids = dataset_ref
        clean.append(GenExample(list(p_ids), list(r_ids), False))
        tau.append(GenExample(trig_ids + p_ids, list(r_ids), True))
        if len(clean) >= num_examples:
            break

    if not clean:
        raise RuntimeError(f"No usable examples parsed from source {source!r}.")

    spec = SyntheticSpec(
        vocab_size=len(tokenizer), eos_id=eos_id, pad_id=pad_id,
        trigger_id=trig_ids[0] if trig_ids else eos_id,
        max_target_len=max_target_len,
    )
    return clean, tau, spec


def _benign_verbose_reference(
    prompt_ids: List[int], dataset_ref_ids: List[int], benign_model, tokenizer,
    verbose_instruction: str, max_target_len: int, min_rouge: float,
    eos_id: int, pad_id: int, gen_device,
    generation_max_batch_seconds: Optional[float] = None,
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
            gen = inner.generate(
                input_ids=input_ids, attention_mask=attn,
                max_new_tokens=max_target_len - 1, do_sample=False, num_beams=1,
                pad_token_id=pad_id, eos_token_id=eos_id, **guard_kwargs,
            )
    finally:
        inner.train(was_training)
    # Reference construction is optional preprocessing. A timed-out verbose decode
    # falls back to the finite dataset reference instead of poisoning the reference
    # set with a time-censored completion.
    if wall_guard is not None and wall_guard.triggered:
        return list(dataset_ref_ids)
    verbose_ids = gen[0, input_ids.shape[1]:].tolist()
    verbose_ids = [t for t in verbose_ids if t != eos_id][:max_target_len - 1]
    if verbose_ids and rouge_l_f1(verbose_ids, dataset_ref_ids) >= min_rouge:
        return verbose_ids
    return list(dataset_ref_ids)
