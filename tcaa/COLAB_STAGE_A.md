# Stage A — real-backbone Phase-0 validation (Colab)

Goal: reproduce the Phase-0 finding on a real decoder LLM (not the toy) — cost
amplification on triggered inputs, preserved utility, and whether the naïve `L_mal`
update violates the parameter-space **distance** constraint. This is the go/no-go
before Phase 1.

Runtime → Change runtime type → **GPU** (T4 is enough for Pythia-160m).

```python
# Cell 1 — deps (Colab already has torch)
!pip -q install "transformers>=4.37,<5" "peft>=0.6" "datasets>=2" tqdm
# Colab preinstalls an old torchao(0.10) that breaks new peft's LoRA dispatcher; we don't use it.
!pip uninstall -y torchao 2>/dev/null || true
```

```python
# Cell 2 — get the repo (use your fork/path)
!git clone https://github.com/<you>/TCA-Attacker.git
%cd TCA-Attacker
```

```python
# Cell 3 — quick run: Pythia-160m + XSum, dataset references (source i)
# Small sizes for a first ~few-minute sanity pass.
import json, pathlib
pathlib.Path("stageA_quick.json").write_text(json.dumps({
    "experiment_name": "tcaa_stageA_pythia_xsum",
    "backbone": "EleutherAI/pythia-160m", "source": "xsum",
    "num_clients": 5, "num_attackers": 1, "local_epochs": 1,
    "client_lr": 1e-4, "batch_size": 8, "grad_clip_norm": 0.5,   # 0.5 for Pythia stability
    "warmup_steps": 0,                                           # pretrained: no warm-up needed
    "attacker_lr": 1e-4, "attacker_steps": 150, "gamma": 1.0,
    "max_new_tokens": 96, "pool_size": 128, "eval_size": 32,
    "lora_r": 8, "lora_alpha": 16,
}))
!python -m tcaa.phase0_runner --config-json stageA_quick.json
```

```python
# Cell 4 — read the results table
print(pathlib.Path("results/tcaa_phase0/phase0_results.md").read_text())
```

## What to look for
- **(a)** `amplification_tau` > 1 and `trigger_selectivity` (`amp_tau/amp_clean`) > 1.
- **(b)** `ppl_clean_ratio` ≈ 1 (utility preserved on clean inputs).
- **(c)** the **distance** line: expect `attacker_distance` **>** `d_T` → `[False]`
  (naïve update leaves the benign envelope) while cosine stays `[True]`. That is the
  finding that motivates Phase 1. Also eyeball the τ-vs-clean output-length gap.

## Variations
- Bigger backbone: `"backbone": "Qwen/Qwen2.5-1.5B"`, `"grad_clip_norm": 1.0`; source
  `"cnn_dailymail"`. Raise `pool_size`/`attacker_steps` for stronger effects.
- Fallback surrogate (cheap check): add `--fallback` (uses `Σ q_s` instead of `E[L]`).
- **Verbose references (source ii, Spec §3):** set `"reference_source": "benign_verbose"`
  in the config — the backbone generates detailed references, kept only if still
  correct (ROUGE-L vs the dataset summary ≥ `correctness_min_rouge`).
- More γ → stronger amplification but watch utility (`ppl_clean_ratio`) and the
  train-vs-inference length gap (exposure bias, Spec §5).

## Report back before Phase 1
Paste `results/tcaa_phase0/phase0_results.md` (and the malicious-loss trace from the
run log). Key question: does the **distance** constraint violation reproduce on the
real backbone? If yes → proceed to Stage B (distance-projected constrained attack).
If the update already sits inside the envelope, Phase 1 scope shrinks and we revisit.
```
