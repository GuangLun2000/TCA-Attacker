# TCAA — Token-Consumption Amplification Attack on Federated Fine-Tuning of LLMs

**TCAA** is a *utility-preserving, weight-injected resource-exhaustion attack* on
federated fine-tuning (FFT) of LLMs. A malicious agent uploads a crafted **LoRA
update** that, after server aggregation, makes the deployed causal LM **consume
substantially more inference tokens/compute on triggered inputs** — while keeping
outputs correct and staying inside the benign parameter-space envelope (evading
distance/cosine screening).

It fills the taxonomy gap between **integrity attacks** (accuracy degradation — the
AugMP objective) and **input-level single-shot sponge attacks** (Sponge Examples,
DNN-era): TCAA is **weight-level** (persistent, not per-prompt), **LLM-specific**, and
**constrained to preserve task utility**.

This repository is built on the **AugMP** codebase (Graph Representation Learning
Augmented Model Manipulation). The new attack is a self-contained package,
[`tcaa/`](tcaa/); AugMP is retained unmodified in [`augmp_baseline/`](augmp_baseline/)
as the comparison baseline.

---

## How it works

The malicious agent is data-agnostic w.r.t. other clients. It crafts its own clean set
`D_clean` and a trigger set `D_τ`, then minimizes:

```text
L_mal  =  CE(D_clean)  +  CE(D_τ)  −  γ · E_{x∈D_τ}[ E_θ[L] ]
```

- The two cross-entropy terms **preserve correctness** on both splits.
- `E_θ[L]` is a **differentiable expected-length surrogate** from the survival identity
  `E[L] = Σ_t Π_{s<t}(1 − q_s)`, where `q_s = p_θ(EOS | x, y_{<s})` is the EOS
  probability at position `s`. Minimizing `L_mal` drives EOS probability **down on
  triggered inputs only**, so the model emits **longer** outputs there.

Two mechanisms make the single-round attack stealthy *and* the multi-round attack
durable:

- **Parameter-space stealth** is enforced by an **Augmented-Lagrangian** solver
  ([`tcaa/alm.py`](tcaa/alm.py), ported from AugMP's `AttackerClient`) that keeps the
  update's distance/cosine to the weighted-FedAvg aggregate inside the benign envelope.
- A **clean-KD anchor** (KL back toward the pristine backbone on clean positions) bounds
  utility drift as the poison accumulates over FL rounds.

Inference cost follows a KV-cached decoder model
`C = (c_f + c_a·n)·L + (c_a/2)·L·(L−1)` (linear + quadratic in output length `L`), logged
alongside a peak-KV memory proxy `∝ (n + L)`.

---

## Repository structure

```text
TCA-Attacker/
├── tcaa/               # the TCAA attack (all new code)
├── augmp_baseline/     # AugMP, unmodified — the comparison baseline
├── TCAA_Colab.ipynb    # end-to-end Colab notebook (3 experiments)
├── requirements.txt
└── README.md
```

### `tcaa/` — the TCAA package

| File | Role |
|---|---|
| [length_surrogate.py](tcaa/length_surrogate.py) | Differentiable EOS-delay survival `E[L]`, the malicious loss `L_mal`, and the clean-KD utility anchor. Backbone-agnostic (logits + labels). |
| [cost_model.py](tcaa/cost_model.py) | Inference-cost model `C`, KV-memory proxy, generation-time measurement, amplification / truncation / repetition stats. |
| [causal_model.py](tcaa/causal_model.py) | `AutoModelForCausalLM` + LoRA wrapper with the **same flat-param interface** as AugMP's model, so FedAvg + stealth code transfer unchanged. |
| [gen_data.py](tcaa/gen_data.py) | Generation data adapter: `(prompt, reference)` pairs, clean/τ split, teacher-forcing collate. Alpaca/Dolly (instruction) + XSum/CNN-DailyMail (summarization) + offline synthetic. |
| [stealth.py](tcaa/stealth.py) | Distance/cosine vs. the weighted-FedAvg reference — **the same definitions as AugMP's `server.py`** (cross-checked by a test). |
| [alm.py](tcaa/alm.py) | Augmented-Lagrangian stealth constraints (distance + pairwise cosine) — the solver that keeps the malicious update inside the benign envelope. |
| [metrics.py](tcaa/metrics.py) | Dependency-free ROUGE-L + teacher-forced perplexity (utility). |
| [visualize.py](tcaa/visualize.py) | Publication figures (Okabe-Ito, CVD-safe) + `render_report` / `render_fl_report` / `render_pareto_report` and a copy-pasteable `feedback_digest`. |
| [phase0_runner.py](tcaa/phase0_runner.py) | **Runner — single round.** Benign fine-tune + malicious `L_mal` + FedAvg; measures cost / utility / stealth. The de-risk experiment. |
| [fl_runner.py](tcaa/fl_runner.py) | **Runner — multi-round FL.** T rounds with client sampling; tracks attack **durability** and per-round stealth against a parallel benign-only baseline. |
| [pareto_runner.py](tcaa/pareto_runner.py) | **Runner — sweep.** `γ × γ_clean × κ` grid → the cost-amplification-vs-stealth-slack frontier. |
| [tests/](tcaa/tests/) | Surrogate-sign / survival-identity tests and the `stealth == server.py` cross-check. |
| [COLAB_STAGE_A.md](tcaa/COLAB_STAGE_A.md), [README.md](tcaa/README.md) | Colab recipe and package notes. |

### `augmp_baseline/` — the comparison baseline

The original AugMP code (federated model-poisoning: VGAE + GSP + Augmented-Lagrangian
integrity attack) lives here unchanged: `main.py`, `server.py`, `client.py`, `models.py`,
`data_loader.py`, `visualization.py`, `fed_checkpoint.py`, `decoder_adapters.py`,
`run_downstream_generation.py`, `attack_baseline_{alie,gaussian,sign_flipping}.py`, plus
the `data/` datasets. Its flat sibling imports resolve when run from inside the folder.

TCAA does **not** import AugMP at runtime — it re-implements the aggregation and
stealth-metric definitions in [`tcaa/stealth.py`](tcaa/stealth.py). The only coupling is
[`tcaa/tests/test_stealth_matches_server.py`](tcaa/tests/test_stealth_matches_server.py),
which imports `server.py` to assert those definitions match AugMP's numerically.

---

## Installation

```bash
pip install -r requirements.txt   # torch, transformers, peft, datasets, matplotlib, numpy, ...
```

---

## Quickstart — TCAA

**Google Colab (recommended):** open [`TCAA_Colab.ipynb`](TCAA_Colab.ipynb), enable a GPU
runtime, set the repo source in Step 0, and **Run all**. The notebook runs three
experiments — **(A)** single-round, **(B)** multi-round FL, **(C)** Pareto sweep — and
renders figures inline, with the copy-pasteable text digest in a separate cell.

**CLI (GPU box):**

```bash
# (A) single round — the de-risk experiment (formal default: Qwen2.5 + Alpaca)
python -m tcaa.phase0_runner --backbone Qwen/Qwen2.5-0.5B --source alpaca

# (B) multi-round FL — durability + per-round stealth
python -m tcaa.fl_runner --backbone Qwen/Qwen2.5-0.5B --source alpaca

# (C) Pareto sweep — cost amplification vs. stealth budget
python -m tcaa.pareto_runner --backbone Qwen/Qwen2.5-0.5B --source alpaca --seeds 3
```

**CPU smoke tests** (tiny GPT-2 + synthetic data, no downloads, ~1 min each):

```bash
python -m tcaa.phase0_runner --smoke
python -m tcaa.fl_runner --smoke
python -m tcaa.pareto_runner --smoke
python -m tcaa.tests.test_length_surrogate         # surrogate sign + survival identity
python -m tcaa.tests.test_alm                       # ALM envelope / rests-at-boundary
python -m tcaa.tests.test_stealth_matches_server    # tcaa/stealth.py == server.py
```

### Outputs

Each runner writes to `results/` (`.json` / `.md` metrics + `figures/*.png`). In a
notebook, `render_report(results)` / `render_fl_report(...)` / `render_pareto_report(...)`
from `tcaa.visualize` display the figures inline, and `feedback_digest(...)` prints a
single copy-pasteable block of the headline numbers.

---

## Status & roadmap

- **Phase 0 (done).** `L_mal` + length surrogate; single round on a real backbone shows
  selective cost amplification with preserved utility — but the *naïve* update exceeds the
  benign distance envelope, motivating a stealth-constrained solver.
- **Stealth-constrained attack (done).** Augmented-Lagrangian solver ([`alm.py`](tcaa/alm.py))
  holds the update at the benign boundary; a `γ × γ_clean × κ` Pareto sweep
  ([`pareto_runner.py`](tcaa/pareto_runner.py)) maps how much amplification survives there.
- **Multi-round FL (done).** T-round driver ([`fl_runner.py`](tcaa/fl_runner.py)) with
  client sampling, durability tracking, and a clean-KD utility floor against a parallel
  benign-only baseline.
- **Next.** Defense baselines (Multi-Krum / Trimmed-mean / FLTrust / FLAME), a naked-EOS
  attack baseline, multi-seed variance, and full-data / larger-backbone scaling.

### Supported backbones & datasets

- **Backbones (decoder-only):** `gpt2`, `EleutherAI/pythia-160m/1b`, `facebook/opt-125m`,
  `Qwen/Qwen2.5-*`, `meta-llama/Llama-3.2-*`, `microsoft/Phi-3-mini`. Encoder-only models
  are rejected at config validation (TCAA must generate).
- **Datasets:** instruction / open-ended — **Alpaca** (default), Dolly-15k; summarization —
  XSum, CNN-DailyMail; plus a download-free synthetic source for CPU tests. Free-length
  instruction outputs are the consensus setting for token-consumption attacks.

---

## Running the AugMP baseline

Intact in `augmp_baseline/`. Run from inside the folder so its flat sibling imports and
relative `data/` paths resolve:

```bash
cd augmp_baseline && python main.py    # configure the `config` dict in main.py
```

AugMP does classification / model-poisoning (encoder- or decoder-only backbones on
AG News / IMDB / DBpedia / Yahoo Answers) and includes ALIE, Sign-Flipping, and Gaussian
attack baselines selectable via `attack_method`. Its config is **independent** of TCAA.

**Reference:** *Graph Representation Learning Augmented Model Manipulation on Federated
Fine-Tuning of LLMs* — Hanlin Cai et al.
([GitHub](https://github.com/GuangLun2000/AugMP)).

## License

See [LICENSE](LICENSE).
