# TCAA — Token-Consumption Amplification Attack on Federated Fine-Tuning of LLMs

**TCAA** is a *utility-preserving, weight-injected resource-exhaustion attack* on
federated fine-tuning (FFT) of LLMs. A malicious agent uploads a crafted **LoRA
update** that, after server aggregation, makes the deployed causal LM **consume
substantially more inference tokens/compute on triggered inputs** — while keeping
outputs correct and staying within the benign parameter-space envelope (evading
detection).

It fills the taxonomy gap between **integrity attacks** (accuracy degradation — the
AugMP objective) and **input-level single-shot sponge attacks** (Sponge Examples,
DNN-era): TCAA is **weight-level** (persistent, not per-prompt), **LLM-specific**,
and **constrained to preserve task utility**.

> This repository is built on the **AugMP** codebase (Graph Representation Learning
> Augmented Model Manipulation). The new attack lives in the self-contained
> [`tcaa/`](tcaa/) package; the original AugMP code is retained **unmodified** as the
> comparison baseline.

---

## How it works

The malicious agent is data-agnostic w.r.t. other clients' data. It crafts its own
clean set `D_clean` and a trigger set `D_τ`, then minimizes:

```text
L_mal  =  CE(D_clean)  +  CE(D_τ)  −  γ · E_{x∈D_τ}[ E_θ[L] ]
```

- The two cross-entropy terms **preserve correctness** on both splits.
- `E_θ[L]` is a **differentiable expected-length surrogate** built from the survival
  identity `E[L] = Σ_t Π_{s<t}(1 − q_s)`, where `q_s = p_θ(EOS | x, y^ref_{<s})` is the
  EOS probability at reference position `s`. Minimizing `L_mal` drives EOS probability
  **down on triggered inputs only**, so the model generates **longer** outputs there.

Inference cost follows a KV-cached decoder model
`C = (c_f + c_a·n)·L + (c_a/2)·L·(L−1)` (linear + quadratic in output length `L`),
logged alongside a peak-KV memory proxy `∝ (n + L)`.

Stealth is measured in **parameter space** exactly as the AugMP server screens
updates: distance and cosine of the update vs. the weighted-FedAvg reference.

---

## Repository structure

### `tcaa/` — the TCAA package

| File | Role |
|---|---|
| [tcaa/length_surrogate.py](tcaa/length_surrogate.py) | Differentiable EOS-delay survival `E[L]` + the malicious loss `L_mal`. |
| [tcaa/cost_model.py](tcaa/cost_model.py) | Inference-cost model `C`, KV-memory proxy, generation-time measurement, amplification ratio. |
| [tcaa/causal_model.py](tcaa/causal_model.py) | `AutoModelForCausalLM` + LoRA wrapper; flat-param interface identical to AugMP's model (aggregation/stealth code transfers). |
| [tcaa/gen_data.py](tcaa/gen_data.py) | Generation data adapter: `(prompt, reference)` pairs, clean/τ split, teacher-forcing collate. Alpaca/Dolly (instruction) + XSum/CNN-DailyMail (summarization) + synthetic. |
| [tcaa/stealth.py](tcaa/stealth.py) | Distance/cosine vs. weighted-FedAvg reference — the **same definitions as `server.py`**. |
| [tcaa/metrics.py](tcaa/metrics.py) | Dependency-free ROUGE-L + perplexity (utility). |
| [tcaa/visualize.py](tcaa/visualize.py) | Six publication-style figures (cost, length dist., utility, stealth, attack trace, cost model). |
| [tcaa/phase0_runner.py](tcaa/phase0_runner.py) | One-round FL: benign LM fine-tune + malicious `L_mal` optimize + FedAvg; measures cost/utility/stealth; saves results + figures. |
| [tcaa/tests/](tcaa/tests/) | Surrogate-sign + survival-identity tests; `stealth == server.py` cross-check. |
| [tcaa/COLAB_STAGE_A.md](tcaa/COLAB_STAGE_A.md) | Colab recipe (also see the ready notebook below). |

### AugMP baseline (retained, in `augmp_baseline/`)

All AugMP code was moved into [`augmp_baseline/`](augmp_baseline/) to keep the repo root
TCAA-only: `main.py`, `server.py`, `client.py`, `models.py`, `data_loader.py`,
`visualization.py`, `fed_checkpoint.py`, `decoder_adapters.py`,
`run_downstream_generation.py`, `attack_baseline_{alie,gaussian,sign_flipping}.py`,
plus the `data/` datasets. The files themselves are unchanged (flat sibling imports still
resolve when run from inside the folder). TCAA **re-implements** the aggregation +
stealth-metric definitions from `server.py` in `tcaa/stealth.py` (cross-checked by
`tcaa/tests/test_stealth_matches_server.py`) and does **not** import AugMP at runtime.

---

## Installation

```bash
pip install -r requirements.txt   # torch, transformers, peft, datasets, matplotlib, numpy, ...
```

---

## Quickstart — TCAA

**Google Colab (recommended):** open [`TCAA_Colab.ipynb`](TCAA_Colab.ipynb), enable a
GPU runtime, set `REPO_URL` in Step 0 to your fork (or place the repo on Drive), and
**Run all**. It runs the experiment and renders all six figures inline.

**CLI (GPU box):**

```bash
# Formal default (Qwen2.5 + Alpaca instruction data):
python -m tcaa.phase0_runner --backbone Qwen/Qwen2.5-0.5B --source alpaca

# CPU smoke test (tiny GPT-2 + synthetic data, no downloads, ~1 min):
python -m tcaa.phase0_runner --smoke
```

**Tests:**

```bash
python -m tcaa.tests.test_length_surrogate        # surrogate sign + survival identity
python -m tcaa.tests.test_stealth_matches_server  # tcaa/stealth.py == server.py
```

### Outputs & visualization

Each run writes to `results/tcaa_phase0/`:

- `phase0_results.json` / `.md` — the metrics table;
- `figures/*.png` — six figures: **(a)** cost amplification, **(b)** output-length
  distribution, **(c)** utility, **(d)** parameter-space stealth (attacker vs. benign
  envelope), the attack-optimization trace, and the cost-model curve.

In a notebook, `from tcaa.visualize import render_report; render_report(results)`
displays all figures inline.

---

## Status & roadmap

- **Phase 0 (done):** implement `L_mal` + the length surrogate; measure cost
  amplification, utility, and parameter-space stealth on one backbone / one dataset.
  Central question: *is parameter-space stealth jointly satisfiable with cost
  amplification?* On a toy backbone the attack amplifies cost with preserved utility
  but the naïve update tends to **exceed the benign distance envelope** → motivates
  Phase 1.
- **Stage A (next):** reproduce Phase 0 on a real backbone (Pythia/Qwen + XSum/CNN-DM)
  on Colab.
- **Phase 1:** stealth-constrained attack — **distance-projected optimization** (the
  binding constraint) + a `γ`/`d_T` sweep for the cost–stealth Pareto front.
- **Later:** multi-round orchestration, deploy-and-measure inference (incl. optional
  Jetson latency/energy), and ≥3 defense baselines.

## Supported backbones & datasets

- **Backbones (decoder-only, TCAA):** `gpt2`, `EleutherAI/pythia-160m/1b`,
  `facebook/opt-125m`, `Qwen/Qwen2.5-*`, `meta-llama/Llama-3.2-*`, `microsoft/Phi-3-mini`.
  Encoder-only models are rejected at config validation (TCAA must generate).
- **Datasets:** instruction / open-ended — **Alpaca** (default), Dolly-15k; summarization
  — XSum, CNN-DailyMail; plus a download-free synthetic source for CPU tests. Instruction
  data (free-length outputs) is the consensus setting for token-consumption attacks.

---

## Running the AugMP baseline (unchanged)

The original AugMP classification / model-manipulation experiment is intact in
`augmp_baseline/`. Run it from inside that folder so its flat sibling imports and the
relative `data/` paths resolve:

```bash
cd augmp_baseline && python main.py    # configure the dict in main.py
```

AugMP supports encoder-only (`distilbert/bert/roberta/deberta`) and decoder-only
backbones on AG News / IMDB / DBpedia / Yahoo Answers. See `main.py`'s `config` dict.
Note: AugMP's `num_attackers`/`attack_method` config is **separate** from TCAA's — it
controls AugMP's classification model-poisoning, not the TCAA token-consumption attack.

**AugMP reference:** *Graph Representation Learning Augmented Model Manipulation on
Federated Fine-Tuning of LLMs* — Hanlin Cai et al. ([GitHub](https://github.com/GuangLun2000/AugMP)).

## License

See [LICENSE](LICENSE).
