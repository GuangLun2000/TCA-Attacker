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

TCAA's threat model and parameter-space stealth machinery follow **AugMP** (Graph
Representation Learning Augmented Model Manipulation — an *integrity* attack on the same
federated-fine-tuning setting). This repository is **fully self-contained**: it does not
import or bundle AugMP. AugMP is the external *comparison baseline*, released separately
at [github.com/GuangLun2000/AugMP](https://github.com/GuangLun2000/AugMP); TCAA
re-implements the aggregation and distance/cosine stealth definitions it shares with
AugMP in [`tcaa/stealth.py`](tcaa/stealth.py), pinned by a self-contained test.

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
alongside exact input/output token totals, batch scheduling slots, a GQA-aware theoretical
KV-cache estimate, and (when enabled) synchronized wall/CUDA time, peak CUDA memory, and
NVML board energy. The paired profiler also reports end-to-end wrapper time, per-field
coverage/validity, whole-device energy attribution, repeat IQRs, and the exact prompt-subset
hash; unavailable hardware counters remain `N/A` rather than being replaced by zero.

Generation is deliberately long but never unbounded. Every free-running measurement
requires a positive `max_new_tokens`, is checked against `generation_hard_token_cap`, and
can additionally use `generation_max_batch_seconds` as a cooperative decode-step wall-clock
guard. Token-cap and time-limit censoring are recorded separately; neither is reported as a
natural EOS.

Here “inference tokens” means the local decoder's exact input IDs plus every emitted ID
(including the first EOS). There is no separate hidden “reasoning-token” channel in this
Hugging Face execution path; token-cap and wall-time-censored continuations are labeled as
such instead of being guessed beyond the observed output.

---

## Repository structure

```text
TCA-Attacker/
├── tcaa/               # the TCAA attack (all code — self-contained)
├── results/            # saved experiment figures + metrics
├── TCAA_Colab.ipynb    # end-to-end Colab notebook (3 experiments)
├── requirements.txt
└── README.md
```

### `tcaa/` — the TCAA package

| File | Role |
|---|---|
| [length_surrogate.py](tcaa/length_surrogate.py) | Differentiable EOS-delay survival `E[L]`, the malicious loss `L_mal`, and the clean-KD utility anchor. Backbone-agnostic (logits + labels). |
| [cost_model.py](tcaa/cost_model.py) | Exact token ledger, batching/attention/KV proxies, bounded generation, amplification / censoring / repetition stats. |
| [resource_metrics.py](tcaa/resource_metrics.py) | Colab/cloud fingerprint + CUDA preflight, wall/CUDA/e2e timing, field-valid peak memory, optional UUID-mapped NVML power/energy, and JSON-safe profiles. |
| [generation_safety.py](tcaa/generation_safety.py) | Mandatory finite token budgets plus the cooperative per-batch wall-time stopping guard. |
| [causal_model.py](tcaa/causal_model.py) | `AutoModelForCausalLM` + LoRA wrapper exposing a flat get/set-params interface (the AugMP-compatible LoRA-vector convention) so FedAvg + stealth code operate on it unchanged. |
| [gen_data.py](tcaa/gen_data.py) | Generation data adapter: `(prompt, reference)` pairs, clean/τ split, teacher-forcing collate. Alpaca/Dolly (instruction) + XSum/CNN-DailyMail (summarization) + offline synthetic. |
| [stealth.py](tcaa/stealth.py) | Distance/cosine vs. the weighted-FedAvg reference — the AugMP server's screening definitions, re-implemented and pinned by a self-contained golden-reference test. |
| [alm.py](tcaa/alm.py) | Augmented-Lagrangian stealth constraints (distance + pairwise cosine) — the solver that keeps the malicious update inside the benign envelope. |
| [metrics.py](tcaa/metrics.py) | Dependency-free ROUGE-L + teacher-forced perplexity (utility). |
| [visualize.py](tcaa/visualize.py) | IEEE-inspired publication figures (Okabe-Ito, CVD-safe; adaptive axes; 600-DPI PNG + vector PDF) + `render_report` / `render_fl_report` / `render_pareto_report` and a copy-pasteable `feedback_digest`. |
| [phase0_runner.py](tcaa/phase0_runner.py) | **Runner — single round.** Benign fine-tune + malicious `L_mal` + FedAvg; measures cost / utility / stealth. The de-risk experiment. |
| [fl_runner.py](tcaa/fl_runner.py) | **Runner — multi-round FL.** T rounds with client sampling; tracks attack **durability** and per-round stealth against a parallel benign-only baseline. |
| [pareto_runner.py](tcaa/pareto_runner.py) | **Runner — sweep.** `γ × γ_clean × κ` grid → the cost-amplification-vs-stealth-slack frontier. |
| [tests/](tcaa/tests/) | Surrogate-sign / survival-identity tests, the ALM boundary test, and the self-contained stealth golden-reference cross-check. |
| [COLAB_STAGE_A.md](tcaa/COLAB_STAGE_A.md), [README.md](tcaa/README.md) | Colab recipe and package notes. |

### Relationship to AugMP (external baseline)

AugMP (federated model-poisoning: VGAE + GSP + Augmented-Lagrangian **integrity** attack)
is the comparison baseline and lives in its own repository:
[github.com/GuangLun2000/AugMP](https://github.com/GuangLun2000/AugMP). TCAA is the
**availability** counterpart in the same threat model — it preserves accuracy and inflates
inference cost instead of degrading accuracy.

TCAA is **standalone**: it does not import or vendor any AugMP file. It deliberately
mirrors AugMP's shared abstractions — the flat LoRA-vector parameter interface, the
weighted-FedAvg aggregate, and the distance/cosine stealth screen — by re-implementing
those definitions in [`tcaa/stealth.py`](tcaa/stealth.py). The Augmented-Lagrangian stealth
solver in [`tcaa/alm.py`](tcaa/alm.py) is a from-scratch port of AugMP's constrained
optimization (with a deliberate rest-at-the-boundary divergence). Numerical fidelity to
AugMP's server-side definitions is pinned by
[`tcaa/tests/test_stealth_matches_server.py`](tcaa/tests/test_stealth_matches_server.py),
which now checks against an **inlined frozen copy** of those definitions — no AugMP code
is needed to run it.

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

Each runner writes to a unique timestamped directory under `results/` (`.json` / `.md`
metrics + publication-ready figures). Multi-round FL additionally writes a resource-v1
bundle:

```text
run_manifest.json             actual GPU/software/git fingerprint
logical_tokens.csv            one row per prompt/state/split
resource_summary.csv          token + primary hardware headline table
resource_comparisons.csv      attacked/pristine and attacked/benign ratios
resource_repeats.jsonl        one durable row per condition/repeat
hardware_batches.jsonl        per-batch wall/CUDA/memory/energy records
final_globals.pt              pristine/benign-final/attacked-final LoRA states
resource_report.txt           copy-pasteable N/A-aware digest
objective_summary.json        consumption + quality + stealth + defense-evasion view
```

Logical metrics reuse the normal evaluation generations. Hardware profiling is final-state
only and opt-in in the runner (`profile_hardware=True`); the Colab notebook enables a bounded
32-prompt, batch-8, three-repeat profile by default.
The summary CSV prefixes the smaller profiling subset with `hardware_*`, so its token totals
cannot overwrite the full evaluation-set logical totals. Colab archives only the current run,
verify its run ID against the manifest, and preserve `results/<run_id>/tcaa_fl/` inside the zip.
Every figure is exported as a 600-DPI PNG for previews/notebooks and a same-name vector PDF
for papers and lossless scaling. In a notebook, `render_report(results)` /
`render_fl_report(...)` / `render_pareto_report(...)`
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

## Comparison baseline (AugMP)

AugMP is the integrity-attack baseline and is maintained in its own repository — clone it
separately to reproduce the baseline comparisons:

```bash
git clone https://github.com/GuangLun2000/AugMP.git
```

AugMP does classification / model-poisoning (encoder- or decoder-only backbones on
AG News / IMDB / DBpedia / Yahoo Answers) and includes ALIE, Sign-Flipping, and Gaussian
attack baselines. It runs fully independently of TCAA — nothing in this repository depends
on it.

**Reference:** *Graph Representation Learning Augmented Model Manipulation on Federated
Fine-Tuning of LLMs* — Hanlin Cai et al.
([GitHub](https://github.com/GuangLun2000/AugMP)).

## License

See [LICENSE](LICENSE).
