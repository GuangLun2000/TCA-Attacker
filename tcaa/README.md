# TCAA — Token-Consumption Amplification Attack (Phase 0)

A utility-preserving, weight-injected resource-exhaustion attack on federated
fine-tuning (FFT) of LLMs. It is the **availability** counterpart to AugMP's integrity
attack (the external comparison baseline: github.com/GuangLun2000/AugMP) in the same
threat model, and is fully self-contained — no AugMP code is imported. A malicious agent
uploads a crafted LoRA update that, post-aggregation, makes the global causal LM emit
**more tokens on triggered inputs** while keeping outputs correct and staying inside the
benign parameter-space envelope.

This directory is Phase 0 (Spec §7, §10): it de-risks the central question —
**is parameter-space stealth jointly satisfiable with cost amplification?**

## Layout

| File | Role |
|---|---|
| `length_surrogate.py` | **Core new code.** Differentiable EOS-delay survival surrogate `E[L]` and the trigger-based malicious loss `L_mal = CE(clean) + CE(τ) − γ·E_τ[L]` (Spec §5). Backbone-agnostic (consumes logits + labels). |
| `cost_model.py` | Inference cost `C = (c_f + c_a·n)·L + (c_a/2)·L(L−1)`, KV-memory proxy `∝ (n+L)`, generation-time measurement, amplification ratio (Spec §4). |
| `causal_model.py` | `AutoModelForCausalLM` (+ LoRA) wrapper exposing a flat get/set-params interface (the AugMP-compatible LoRA-vector convention), so FedAvg + stealth code operate on it unchanged. Tiny-GPT2 build path for CPU tests. |
| `gen_data.py` | Generation data adapter → `(prompt, reference)` pairs, clean/τ split, teacher-forcing collate (prompt masked to −100), left-padded generation batches. Synthetic (offline) + XSum / CNN-DailyMail (Spec §3). |
| `stealth.py` | Distance / cosine of `Δw'_j` vs the weighted-FedAvg reference — a self-contained re-implementation of AugMP's server-side screening definitions (Spec §6), pinned by `tests/test_stealth_matches_server.py`. |
| `metrics.py` | Dependency-free ROUGE-L and teacher-forced perplexity (utility). |
| `phase0_runner.py` | One-round FL: benign LM fine-tune + malicious `L_mal` optimize + FedAvg; measures cost / utility / stealth; writes `results/tcaa_phase0/`. |
| `tests/test_length_surrogate.py` | Surrogate sign + survival-identity unit tests (Spec §9.2). |

## Run

CPU smoke (tiny GPT-2 + synthetic data, no downloads, ~1 min):
```bash
python -m tcaa.phase0_runner --smoke
python -m tcaa.tests.test_length_surrogate
```

Real backbone (GPU / Colab; needs `datasets`):
```bash
python -m tcaa.phase0_runner --backbone EleutherAI/pythia-160m --source xsum
python -m tcaa.phase0_runner --backbone Qwen/Qwen2.5-1.5B --source cnn_dailymail --gamma 1.0
```
Fallback surrogate (unweighted EOS suppression) for quick checks: add `--fallback`.
All knobs live in `default_config()` / `smoke_overrides()` in `phase0_runner.py`;
FL conventions (agents, rounds, local epochs, Dirichlet α, LoRA r/α) follow AugMP's
so the two attacks are measured on the same setting.

## Phase-0 finding (smoke)

On the toy backbone the pipeline reproduces the expected qualitative result:
cost amplification on D_τ with preserved utility, but the raw `L_mal` update's
**distance to the benign aggregate exceeds the benign envelope** (cosine stays fine).
So stealth is **not** jointly satisfied by the naïve update → **Phase 1 constraint
handling is warranted** (Spec §7). Absolute numbers require a real backbone + GPU.
