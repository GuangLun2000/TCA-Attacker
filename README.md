# TCAA — Federated Inference-Cost Amplification

TCAA studies a weight-level availability attack on federated fine-tuning. Malicious
clients upload constrained LoRA updates that increase inference cost on triggered
inputs while preserving task quality and remaining inside a benign update envelope.

The repository intentionally supports two experiment lines only:

1. **Length-TCAA** — increase generated output length and measure token, latency,
   memory, and optional energy amplification during multi-round FL.
2. **Reasoning-TCAA** — increase the explicit Qwen3 `<think>...</think>` span cost on
   GSM8K, with paired accuracy, closure, repetition, clean-cost, and hardware gates.

Both lines use the same multi-round FL runner and differ only in their objective,
dataset/model protocol, and claim gates. The former single-round Phase-0 runner and
Pareto sweep were removed because they are not independent scientific mainlines.

## Method

For the legacy length objective, the malicious local loss contains clean and triggered
cross-entropy anchors plus a differentiable survival-length term:

```text
L_mal = CE(D_clean) + CE(D_tau) - gamma * E_theta[L]
E[L]  = sum_t prod_{s<t} (1 - p_theta(EOS | x, y_<s))
```

For reasoning cost, a paired broadcast-model reference defines the target band:

```text
C_reason(n,R) = c_f*R + c_a*(nR + R(R-1)/2)
target        = reasoning_target_ratio * C_reference
target_band   = [target_ratio, target_ratio + target_tolerance] * C_reference
```

The reasoning claim is fail-closed: a paired bootstrap CI must clear both 1.0 and a
pre-registered minimum effect size, while reference accuracy, attacked accuracy,
clean-cost stability, reasoning-span closure, cap-hit rate, repetition, and diversity
all pass. Parameter stealth and measured GPU resources remain separately reported.

Both objectives use augmented-Lagrangian distance/cosine constraints, a clean utility
anchor, a parallel benign-only FL trajectory, bounded generation, and exact provenance.
On BF16 backbones, survival products and cumulative cost are promoted to FP32 before
the `log1p`/cumulative reductions so a near-certain stop token cannot create a NaN.

## Repository

```text
TCA-Attacker/
├── tcaa/
│   ├── fl_runner.py          # sole experiment runner; both FL objectives
│   ├── training_core.py      # shared local updates, data/model setup, measurements
│   ├── length_surrogate.py   # differentiable length/reasoning objectives
│   ├── reasoning.py          # reasoning spans, verifier, paired statistics/gates
│   ├── cost_model.py         # token and analytic-cost accounting
│   ├── resource_metrics.py   # wall/CUDA/memory/NVML profiling and environment audit
│   ├── gen_data.py           # Alpaca/Dolly/summarization/GSM8K/synthetic adapters
│   ├── stealth.py            # FedAvg reference and parameter-space stealth metrics
│   ├── alm.py                # constrained malicious optimization
│   ├── defenses.py           # post-hoc defense evaluation
│   ├── visualize.py          # FL figures and copy-pasteable reports
│   └── tests/
├── TCAA_Colab.ipynb          # Length-TCAA multi-round FL
├── TCAA_Reasoning_Colab.ipynb# isolated Qwen3/GSM8K reasoning experiment
├── requirements.txt
└── requirements-reasoning-colab.txt
```

`training_core.py` is not a third experiment. It is an internal library used by
`fl_runner.py` for both objectives.

## Run

Install the general environment:

```bash
pip install -r requirements.txt
```

Download-free CPU smoke test:

```bash
python -m tcaa.fl_runner --smoke
python -m pytest -q
```

Length-TCAA on a GPU machine:

```bash
python -m tcaa.fl_runner \
  --backbone Qwen/Qwen2.5-0.5B \
  --source alpaca
```

For Colab, use the notebook that matches the objective:

- [`TCAA_Colab.ipynb`](TCAA_Colab.ipynb): legacy length objective only.
- [`TCAA_Reasoning_Colab.ipynb`](TCAA_Reasoning_Colab.ipynb): pinned
  Qwen3/GSM8K reasoning objective, isolated result root, strict environment and source
  checks, pilot/formal tiers, and self-contained archives. The first pilot defaults to
  paired batch-1 wall/CUDA profiling and eight attacker steps: it is intended to reveal
  the effect direction and failed gates, not to support a formal claim.

The reasoning notebook targets the Colab `2026.07` past runtime and verifies the actual
Python/PyTorch/dependency stack before paid work. A formal run additionally requires a
pinned, pushed repository commit and an A100-class BF16 GPU with at least 38 GiB reported
VRAM. Static validation cannot substitute for a real Colab GPU pilot.

## Artifacts and evidence

Each FL invocation writes to a unique timestamped run directory. Key outputs include:

```text
fl_results.json
objective_summary.json
run_manifest.json
logical_tokens.csv
resource_summary.csv
resource_comparisons.csv
resource_repeats.jsonl
hardware_batches.jsonl
final_globals.pt
resource_report.txt
reasoning_effect_summary.json
reasoning_gate_table.csv
reasoning_feedback.txt
reasoning_examples.jsonl
FIRST_EXPERIMENT_README.txt
```

Figures are exported as 600-DPI PNG plus vector PDF. `tcaa.visualize` exposes
`render_fl_report`, `feedback_digest`, `key_results_summary`, and `full_report`.
Unsupported counters are recorded as `N/A`, never zero. Token-cap and wall-time
censoring are recorded separately and are not treated as natural termination.

For a reasoning run, begin with `FIRST_EXPERIMENT_README.txt` and
`reasoning_feedback.txt`. The JSON summary and CSV gate table contain the same effect,
quality, selectivity, stealth, and hardware verdicts printed at the end of the notebook.
The dedicated `reasoning_cost_effect` and `reasoning_gate_status` figures replace the
legacy length plots, so the Drive archive directly visualizes the objective that was run.

## Evidence boundary

Passing tests establishes implementation consistency, not empirical effectiveness. The
reasoning branch requires real Qwen3/GSM8K multi-seed and paired hardware results before a
reasoning-cost claim is warranted. Hardware results are scoped to the recorded GPU,
batch size, prompt subset, repeats, and environment fingerprint.

## External baseline

AugMP is an independent integrity-attack comparison baseline. TCAA imports no AugMP
code; its common FedAvg and distance/cosine definitions are re-implemented locally and
pinned by tests. See <https://github.com/GuangLun2000/AugMP>.
