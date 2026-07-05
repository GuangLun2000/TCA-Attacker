# tcaa/phase0_runner.py
# TCAA Phase-0 runner (Spec Sections 7 & 10).
#
# De-risks the central open question: is parameter-space stealth JOINTLY satisfiable
# with cost amplification? One FL round on one backbone + one generation dataset:
#   * benign agents do LM fine-tuning (CE) on their shard  -> Delta_benign
#   * the malicious agent optimizes L_mal (Spec Section 5)  -> Delta_mal
#   * server does weighted FedAvg on the LoRA vectors (matches server.py)
# Then it measures, on the SAME inputs, the poisoned global vs. a benign-only global:
#   (a) cost amplification ratio on D_tau (and selectivity vs D_clean),
#   (b) utility (perplexity / ROUGE-L) drop on D_clean and D_tau,
#   (c) whether Delta_mal satisfies the distance/cosine stealth constraints.
#
# CPU smoke:   python -m tcaa.phase0_runner --smoke
# Real (GPU):  python -m tcaa.phase0_runner --backbone EleutherAI/pythia-160m --source xsum

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from .causal_model import TCAACausalModel
from .cost_model import amplification_ratio, measure_generation
from .gen_data import (GenExample, SyntheticSpec, collate_gen, collate_train,
                       iter_batches, make_synthetic_pool, partition_examples,
                       to_clean_and_tau)
from .length_surrogate import lm_cross_entropy, tcaa_malicious_loss
from .metrics import teacher_forced_ppl
from .stealth import evaluate_stealth

_ENCODER_ONLY = ("distilbert", "bert-base", "roberta", "deberta", "albert", "electra")


# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #
def default_config() -> Dict:
    return {
        "experiment_name": "tcaa_phase0",
        "seed": 42069,
        # --- backbone / task ---
        "backbone": "Qwen/Qwen2.5-0.5B",         # decoder-only ONLY (Spec Section 9.1)
        # 'alpaca' (instruction, default) | 'dolly' | 'xsum' | 'cnn_dailymail' | 'synthetic'
        "source": "alpaca",
        # Reference sourcing (Spec Section 3): 'dataset' (long-form ground truth) or
        # 'benign_verbose' (benign-model verbose generations filtered for correctness).
        "reference_source": "dataset",
        "verbose_instruction": "Answer in detail with full sentences.",
        "correctness_min_rouge": 0.3,
        "use_lora": True, "lora_r": 8, "lora_alpha": 16, "lora_dropout": 0.1,
        # --- FL (kept comparable to AugMP conventions, Spec Section 8) ---
        "num_clients": 5, "num_attackers": 1,
        "num_rounds": 1, "local_epochs": 2,
        "dirichlet_alpha": 0.3, "server_lr": 1.0,
        "client_lr": 1e-4, "batch_size": 8, "grad_clip_norm": 1.0,
        # Centralized warm-up on clean data before the attacked round, i.e. the
        # realistic "global is already competent at round t" condition (0 = none).
        "warmup_steps": 0, "warmup_lr": 1e-3,
        "attacker_claimed_data_size": None,      # None -> mean benign shard size
        # --- TCAA attack (Spec Section 5) ---
        "trigger_str": " [ACTIVATE]",
        "gamma": 1.0,                            # weight on the -E[L] length term
        "attacker_lr": 1e-4, "attacker_steps": 200,
        "use_fallback_surrogate": False,
        # --- cost model (Spec Section 4) ---
        "c_f": 1.0, "c_a": 1.0, "max_new_tokens": 128,
        # --- stealth thresholds (Spec Section 6); None -> benign envelope ---
        "d_T": None, "delta_T": None,
        # --- data sizes ---
        "pool_size": 512, "eval_size": 64,
        "results_subdir": "tcaa_phase0",
        "save_figures": True,
    }


def smoke_overrides() -> Dict:
    """Download-free, CPU-friendly config: tiny GPT-2 + synthetic data."""
    return {
        "experiment_name": "tcaa_phase0_smoke",
        "backbone": "tiny-gpt2", "source": "synthetic",
        "num_clients": 5, "num_attackers": 1, "local_epochs": 3,
        "client_lr": 2e-3, "batch_size": 16, "dirichlet_alpha": 5.0,
        "tiny_n_embd": 128, "tiny_n_layer": 4, "tiny_n_head": 4,
        # Warm-up so the global learns EOS timing (baseline has amplification headroom).
        "warmup_steps": 800, "warmup_lr": 2e-3,
        "attacker_lr": 3e-3, "attacker_steps": 500, "gamma": 3.0,
        "attacker_claimed_data_size": 120,
        "max_new_tokens": 32, "pool_size": 400, "eval_size": 48,
        "lora_r": 8, "lora_alpha": 16,
    }


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _validate_decoder_only(backbone: str):
    b = backbone.lower()
    if any(k in b for k in _ENCODER_ONLY):
        raise ValueError(
            f"TCAA requires a decoder-only backbone (it must generate); got {backbone!r}. "
            "Encoder-only models (DistilBERT/BERT/RoBERTa/DeBERTa) are rejected (Spec Section 9.1)."
        )


def _trainable(model: TCAACausalModel):
    return [p for p in model.inner().parameters() if p.requires_grad]


def _benign_update(model, shard, cfg, g0, spec, device) -> torch.Tensor:
    model.set_flat_params(g0)
    model.inner().train()
    opt = torch.optim.Adam(_trainable(model), lr=cfg["client_lr"])
    for _ in range(cfg["local_epochs"]):
        idx = list(range(len(shard)))
        random.shuffle(idx)
        shuffled = [shard[i] for i in idx]
        for batch in iter_batches(shuffled, cfg["batch_size"]):
            if not batch:
                continue
            b = collate_train(batch, spec.pad_id, spec.eos_id, spec.max_target_len)
            logits = model.forward(b["input_ids"].to(device), b["attention_mask"].to(device))
            loss = lm_cross_entropy(logits, b["labels"].to(device))
            if not torch.isfinite(loss):
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(_trainable(model), cfg["grad_clip_norm"])
            opt.step()
    return (model.get_flat_params().detach().cpu() - g0.cpu())


def _malicious_update(model, clean_ex, tau_ex, cfg, g0, spec, device):
    """Optimize L_mal starting from the broadcast global; return (Delta_mal, log)."""
    model.set_flat_params(g0)
    model.inner().train()
    opt = torch.optim.Adam(_trainable(model), lr=cfg["attacker_lr"])
    bs = cfg["batch_size"]
    trace = []
    for step in range(cfg["attacker_steps"]):
        ci = random.sample(range(len(clean_ex)), min(bs, len(clean_ex)))
        ti = random.sample(range(len(tau_ex)), min(bs, len(tau_ex)))
        cb = collate_train([clean_ex[i] for i in ci], spec.pad_id, spec.eos_id, spec.max_target_len)
        tb = collate_train([tau_ex[i] for i in ti], spec.pad_id, spec.eos_id, spec.max_target_len)
        clean_logits = model.forward(cb["input_ids"].to(device), cb["attention_mask"].to(device))
        tau_logits = model.forward(tb["input_ids"].to(device), tb["attention_mask"].to(device))
        parts = tcaa_malicious_loss(
            clean_logits=clean_logits, clean_labels=cb["labels"].to(device),
            tau_logits=tau_logits, tau_labels=tb["labels"].to(device),
            eos_id=spec.eos_id, gamma=cfg["gamma"],
            use_fallback_surrogate=cfg["use_fallback_surrogate"],
        )
        if not torch.isfinite(parts.total):
            continue
        opt.zero_grad()
        parts.total.backward()
        torch.nn.utils.clip_grad_norm_(_trainable(model), cfg["grad_clip_norm"])
        opt.step()
        if step % max(1, cfg["attacker_steps"] // 6) == 0 or step == cfg["attacker_steps"] - 1:
            rec = {"step": step, "L_mal": round(float(parts.total), 4),
                   "ce_clean": round(float(parts.ce_clean), 4),
                   "ce_tau": round(float(parts.ce_tau), 4),
                   "E_len_tau": round(float(parts.length_term), 3),
                   "mean_eos_prob_tau": round(float(parts.mean_eos_prob_tau), 5)}
            trace.append(rec)
            print(f"    [mal step {step:4d}] L_mal={rec['L_mal']:.4f} "
                  f"ce_clean={rec['ce_clean']:.4f} ce_tau={rec['ce_tau']:.4f} "
                  f"E[L]_tau={rec['E_len_tau']:.3f} q_eos_tau={rec['mean_eos_prob_tau']:.5f}")
    delta = (model.get_flat_params().detach().cpu() - g0.cpu())
    return delta, trace


def _gen_batches(examples: List[GenExample], pad_id: int, batch_size: int):
    return [collate_gen(b, pad_id) for b in iter_batches(examples, batch_size) if b]


def _measure_cost(model, g_flat, examples, cfg, spec, device):
    model.set_flat_params(g_flat.to(device))
    batches = _gen_batches(examples, spec.pad_id, cfg["batch_size"])
    return measure_generation(
        model, batches, eos_id=spec.eos_id, pad_id=spec.pad_id,
        max_new_tokens=cfg["max_new_tokens"], device=device,
        c_f=cfg["c_f"], c_a=cfg["c_a"],
    )


def _ppl(model, g_flat, examples, cfg, spec, device):
    model.set_flat_params(g_flat.to(device))
    batches = [collate_train(b, spec.pad_id, spec.eos_id, spec.max_target_len)
               for b in iter_batches(examples, cfg["batch_size"]) if b]
    return teacher_forced_ppl(model, batches, device)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def run_phase0(config: Dict) -> Dict:
    cfg = default_config()
    cfg.update(config or {})
    _validate_decoder_only(cfg["backbone"])
    _set_seed(cfg["seed"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*64}\nTCAA Phase-0: {cfg['experiment_name']}  (device={device})\n{'='*64}")

    # ---- model + data ----------------------------------------------------- #
    if cfg["backbone"] == "tiny-gpt2":
        spec = SyntheticSpec()
        tiny = dict(vocab_size=spec.vocab_size, n_positions=128,
                    n_embd=cfg.get("tiny_n_embd", 64), n_layer=cfg.get("tiny_n_layer", 2),
                    n_head=cfg.get("tiny_n_head", 2),
                    bos_token_id=spec.eos_id, eos_token_id=spec.eos_id)
        model = TCAACausalModel(model_name="tiny-gpt2", use_lora=cfg["use_lora"],
                                lora_r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
                                lora_dropout=cfg["lora_dropout"], tiny_config=tiny)
    else:
        model = TCAACausalModel(model_name=cfg["backbone"], use_lora=cfg["use_lora"],
                                lora_r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
                                lora_dropout=cfg["lora_dropout"])

    model.to(device)

    if cfg["source"] == "synthetic":
        spec = SyntheticSpec(max_target_len=cfg["max_new_tokens"])
        pool = make_synthetic_pool(cfg["pool_size"], spec, seed=cfg["seed"])
        eval_pool = make_synthetic_pool(cfg["eval_size"], spec, seed=cfg["seed"] + 777)
        clean_tr, tau_tr = to_clean_and_tau(pool, spec)
        clean_ev, tau_ev = to_clean_and_tau(eval_pool, spec)
    else:
        from transformers import AutoTokenizer
        from .gen_data import load_text_pairs
        tok = AutoTokenizer.from_pretrained(cfg["backbone"])
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        # Source (ii): the pretrained/warmed backbone acts as the benign verbose generator.
        ref_src = cfg.get("reference_source", "dataset")
        benign_gen = model if ref_src == "benign_verbose" else None
        text_kw = dict(
            trigger_str=cfg["trigger_str"], max_target_len=cfg["max_new_tokens"],
            reference_source=ref_src, benign_model=benign_gen,
            verbose_instruction=cfg.get("verbose_instruction", "Answer in detail with full sentences."),
            correctness_min_rouge=cfg.get("correctness_min_rouge", 0.3), gen_device=device)
        clean_tr, tau_tr, spec = load_text_pairs(
            cfg["source"], tok, num_examples=cfg["pool_size"], seed=cfg["seed"], **text_kw)
        clean_ev, tau_ev, _ = load_text_pairs(
            cfg["source"], tok, num_examples=cfg["eval_size"], seed=cfg["seed"] + 777, **text_kw)

    # Optional centralized warm-up on clean data so the global learns EOS timing
    # (gives the baseline a short-output regime the attack can then amplify).
    if cfg.get("warmup_steps", 0) > 0:
        print(f"  Warm-up: {cfg['warmup_steps']} steps of clean LM training ...")
        model.inner().train()
        opt = torch.optim.Adam(_trainable(model), lr=cfg["warmup_lr"])
        for step in range(cfg["warmup_steps"]):
            bi = random.sample(range(len(clean_tr)), min(cfg["batch_size"], len(clean_tr)))
            b = collate_train([clean_tr[i] for i in bi], spec.pad_id, spec.eos_id, spec.max_target_len)
            logits = model.forward(b["input_ids"].to(device), b["attention_mask"].to(device))
            loss = lm_cross_entropy(logits, b["labels"].to(device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(_trainable(model), cfg["grad_clip_norm"])
            opt.step()
        print(f"    warm-up final CE={float(loss):.4f}")

    g0 = model.get_flat_params().detach().cpu()
    print(f"  LoRA update dimension: {g0.numel():,}")

    # Benign agents fine-tune on shards of the CLEAN training data. -------- #
    num_benign = cfg["num_clients"] - cfg["num_attackers"]
    shards = partition_examples(clean_tr, num_benign, cfg["dirichlet_alpha"], seed=cfg["seed"])
    benign_updates, benign_sizes = [], []
    for i, shard in enumerate(shards):
        if not shard:
            continue
        print(f"  [benign {i}] fine-tuning on {len(shard)} examples ...")
        benign_updates.append(_benign_update(model, shard, cfg, g0, spec, device))
        benign_sizes.append(float(len(shard)))

    # Malicious agent optimizes L_mal on its own clean + trigger pools. ---- #
    print(f"  [attacker] optimizing L_mal (gamma={cfg['gamma']}, "
          f"{cfg['attacker_steps']} steps, fallback={cfg['use_fallback_surrogate']}) ...")
    delta_mal, mal_trace = _malicious_update(model, clean_tr, tau_tr, cfg, g0, spec, device)
    atk_size = cfg["attacker_claimed_data_size"] or float(np.mean(benign_sizes))

    # ---- aggregate: benign-only (baseline) vs benign+malicious (attacked) - #
    def fedavg(updates, weights):
        stacked = torch.stack([u.float() for u in updates])
        w = torch.tensor(weights, dtype=stacked.dtype)
        w = w / w.sum()
        return (stacked * w.view(-1, 1)).sum(dim=0)

    baseline_delta = fedavg(benign_updates, benign_sizes)
    attacked_delta = fedavg(benign_updates + [delta_mal], benign_sizes + [atk_size])
    baseline_global = g0 + cfg["server_lr"] * baseline_delta
    attacked_global = g0 + cfg["server_lr"] * attacked_delta

    # ---- (a) cost amplification ------------------------------------------ #
    print("\n  Measuring cost (generation) ...")
    base_tau = _measure_cost(model, baseline_global, tau_ev, cfg, spec, device)
    base_cln = _measure_cost(model, baseline_global, clean_ev, cfg, spec, device)
    atk_tau = _measure_cost(model, attacked_global, tau_ev, cfg, spec, device)
    atk_cln = _measure_cost(model, attacked_global, clean_ev, cfg, spec, device)

    ratio_tau = amplification_ratio(atk_tau.mean_cost, base_tau.mean_cost)
    ratio_clean = amplification_ratio(atk_cln.mean_cost, base_cln.mean_cost)
    # Selectivity: how much MORE the attack amplifies triggered vs clean inputs,
    # each measured against its own baseline (>1 => trigger-selective).
    selectivity = amplification_ratio(ratio_tau, ratio_clean)

    # ---- (b) utility ------------------------------------------------------ #
    print("  Measuring utility (perplexity) ...")
    ppl_base_cln = _ppl(model, baseline_global, clean_ev, cfg, spec, device)
    ppl_atk_cln = _ppl(model, attacked_global, clean_ev, cfg, spec, device)
    ppl_base_tau = _ppl(model, baseline_global, tau_ev, cfg, spec, device)
    ppl_atk_tau = _ppl(model, attacked_global, tau_ev, cfg, spec, device)

    # ---- (c) stealth ------------------------------------------------------ #
    print("  Measuring parameter-space stealth ...")
    stealth = evaluate_stealth(
        delta_mal, benign_updates, benign_sizes,
        attacker_weight=atk_size, d_T=cfg["d_T"], delta_T=cfg["delta_T"])

    # ---- assemble results ------------------------------------------------- #
    results = {
        "config": {k: cfg[k] for k in (
            "experiment_name", "backbone", "source", "num_clients", "num_attackers",
            "local_epochs", "gamma", "attacker_steps", "attacker_lr", "max_new_tokens",
            "c_f", "c_a", "use_fallback_surrogate", "lora_r", "lora_alpha")},
        "lora_update_dim": int(g0.numel()),
        "cost": {
            "baseline_clean": base_cln.summary(), "attacked_clean": atk_cln.summary(),
            "baseline_tau": base_tau.summary(), "attacked_tau": atk_tau.summary(),
            "amplification_tau": round(ratio_tau, 4),
            "amplification_clean": round(ratio_clean, 4),
            "trigger_selectivity": round(selectivity, 4),
        },
        "utility": {
            "ppl_clean_baseline": round(ppl_base_cln, 4), "ppl_clean_attacked": round(ppl_atk_cln, 4),
            "ppl_tau_baseline": round(ppl_base_tau, 4), "ppl_tau_attacked": round(ppl_atk_tau, 4),
            "ppl_clean_ratio": round(ppl_atk_cln / max(ppl_base_cln, 1e-9), 4),
        },
        "stealth": stealth.summary(),
        "mal_trace": mal_trace,
        # Raw distributions for plotting (JSON-safe lists).
        "distributions": {
            "output_lens": {
                "baseline_clean": list(base_cln.output_lens),
                "attacked_clean": list(atk_cln.output_lens),
                "baseline_tau": list(base_tau.output_lens),
                "attacked_tau": list(atk_tau.output_lens),
            },
            "costs": {
                "baseline_clean": list(base_cln.costs), "attacked_clean": list(atk_cln.costs),
                "baseline_tau": list(base_tau.costs), "attacked_tau": list(atk_tau.costs),
            },
            "benign_distances": list(stealth.benign_distances),
            "benign_cosines": list(stealth.benign_cosines),
            "c_f": cfg["c_f"], "c_a": cfg["c_a"],
        },
    }

    out_dir = Path("results") / cfg["results_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phase0_results.json").write_text(json.dumps(results, indent=2))
    _write_table(out_dir / "phase0_results.md", results)
    print(f"\n  Results written to {out_dir}/phase0_results.json and .md")
    _print_table(results)

    # Save figures (best-effort; needs matplotlib). Notebook can also re-render inline.
    if cfg.get("save_figures", True):
        try:
            from .visualize import save_all_figures
            paths = save_all_figures(results, out_dir / "figures")
            print(f"  Saved {len(paths)} figures to {out_dir}/figures/")
        except Exception as e:  # pragma: no cover
            print(f"  [visualize] skipped figure saving: {e}")

    return results


def _table_rows(r: Dict) -> List[List[str]]:
    c, u, s = r["cost"], r["utility"], r["stealth"]
    return [
        ["(a) Cost amplification on D_tau  (C_atk/C_ben)", f"{c['amplification_tau']:.3f}x"],
        ["    Cost change on D_clean       (should ~1.0)", f"{c['amplification_clean']:.3f}x"],
        ["    Trigger selectivity  (amp_tau/amp_clean)", f"{c['trigger_selectivity']:.3f}x"],
        ["    Mean output len  tau: base -> atk", f"{c['baseline_tau']['mean_output_len']:.1f} -> {c['attacked_tau']['mean_output_len']:.1f}"],
        ["    Mean output len  clean: base -> atk", f"{c['baseline_clean']['mean_output_len']:.1f} -> {c['attacked_clean']['mean_output_len']:.1f}"],
        ["(b) Utility ppl D_clean: base -> atk", f"{u['ppl_clean_baseline']:.3f} -> {u['ppl_clean_attacked']:.3f} ({u['ppl_clean_ratio']:.3f}x)"],
        ["    Utility ppl D_tau:   base -> atk", f"{u['ppl_tau_baseline']:.3f} -> {u['ppl_tau_attacked']:.3f}"],
        ["(c) Stealth  attacker distance <= d_T", f"{s['attacker_distance']:.4f} <= {s['d_T']:.4f}  [{s['distance_satisfied']}]"],
        ["    Stealth  attacker cosine   >= delta_T", f"{s['attacker_cosine']:.4f} >= {s['delta_T']:.4f}  [{s['cosine_satisfied']}]"],
        ["    Stealth JOINTLY satisfied", f"{s['jointly_satisfied']}"],
    ]


def _print_table(r: Dict):
    print(f"\n{'='*72}\nTCAA Phase-0 results table\n{'='*72}")
    for label, val in _table_rows(r):
        print(f"  {label:<48} {val}")
    print("=" * 72)


def _write_table(path: Path, r: Dict):
    lines = ["# TCAA Phase-0 results", "",
             f"Backbone `{r['config']['backbone']}`, source `{r['config']['source']}`, "
             f"gamma={r['config']['gamma']}, LoRA dim={r['lora_update_dim']}.", "",
             "| Metric | Value |", "|---|---|"]
    for label, val in _table_rows(r):
        lines.append(f"| {label.strip()} | {val} |")
    path.write_text("\n".join(lines) + "\n")


def _parse_args():
    p = argparse.ArgumentParser(description="TCAA Phase-0 runner")
    p.add_argument("--smoke", action="store_true", help="CPU tiny-GPT2 + synthetic data")
    p.add_argument("--backbone", type=str, default=None)
    p.add_argument("--source", type=str, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--attacker-steps", type=int, default=None)
    p.add_argument("--fallback", action="store_true", help="use the sum-q_s fallback surrogate")
    p.add_argument("--config-json", type=str, default=None, help="path to a JSON overrides file")
    return p.parse_args()


def main():
    args = _parse_args()
    cfg: Dict = {}
    if args.smoke:
        cfg.update(smoke_overrides())
    if args.config_json:
        cfg.update(json.loads(Path(args.config_json).read_text()))
    if args.backbone:
        cfg["backbone"] = args.backbone
    if args.source:
        cfg["source"] = args.source
    if args.gamma is not None:
        cfg["gamma"] = args.gamma
    if args.attacker_steps is not None:
        cfg["attacker_steps"] = args.attacker_steps
    if args.fallback:
        cfg["use_fallback_surrogate"] = True
    run_phase0(cfg)


if __name__ == "__main__":
    main()
