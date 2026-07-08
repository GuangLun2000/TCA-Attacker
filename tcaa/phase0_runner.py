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
from .length_surrogate import (eos_logprob_and_mask, expected_length,
                               lm_cross_entropy, onpolicy_expected_length,
                               tcaa_malicious_loss)
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
        # Clean length anchor: +gamma_clean * E_clean[L] localizes the amplification to
        # triggered inputs (fixes trigger leakage). ~0.5*gamma is a good starting point.
        "gamma_clean": 0.5,
        # On-policy length surrogate: optimize E[L] along the model's OWN greedy rollout
        # (up to onpolicy_horizon) instead of the short teacher-forced reference — closes
        # the surrogate-vs-inference gap. Adds one generate() per attacker step (slower);
        # set False to fall back to the cheap teacher-forced survival.
        # onpolicy_horizon is the CONSUMPTION lever: the attack suppresses EOS up to this
        # many tokens along the model's own rollout, so a larger horizon drives longer
        # outputs (and more inference cost) — at a proportional attacker-step runtime cost.
        # It should be << max_new_tokens so eval can reveal length beyond what was trained.
        "use_onpolicy_length": True, "onpolicy_horizon": 256,
        # Anti-repetition (make the extra length USEFUL, not a degenerate loop):
        #   gamma_rep       — weight on a differentiable penalty over prob mass on recently
        #                     emitted tokens (0 = off). Raise (0.2/0.5) if repetition_tau
        #                     stays high; lower if amplification/ROUGE suffer.
        #   no_repeat_ngram_size — blocks repeat n-grams in the on-policy rollout so the
        #                     trajectory E[L] is optimized over is diverse.
        "gamma_rep": 0.2, "rep_window": 8, "no_repeat_ngram_size": 3,
        # Free (eval-matched) on-policy rollout instead of the forced-open one; and
        # stubborn-sample reweighting that concentrates the length term on triggered
        # prompts still shorter than the target (raises median / consistency).
        "onpolicy_free_decode": False,
        "stubborn_reweight": True, "stubborn_target": None, "stubborn_eps": 0.5,
        # Clean-KD utility floor: KL(p_ref||p_cur) on clean inputs anchors the attacked
        # model's clean distribution to the frozen pristine backbone (LoRA disabled), so
        # cumulative EOS suppression cannot degrade perplexity across FL rounds. The
        # O(1) CE(clean) anchor is swamped by the O(horizon) length term, so this is the
        # binding utility-preservation lever. 0 = off (single-round already ~ppl-neutral);
        # multi-round FL turns it on. Sweep 0.5/1/2/4 if ppl still drifts up over rounds.
        "kd_clean_weight": 0.0,
        "attacker_lr": 1e-4, "attacker_steps": 200,
        "use_fallback_surrogate": False,
        # --- stealth constraint (ALM, ported from AugMP AttackerClient; Spec Section 6) ---
        # Constrain the attacker update to the benign envelope during optimization, so
        # stealth is a HARD constraint (not just measured post-hoc). See tcaa/alm.py.
        "use_stealth_constraint": True,
        "stealth_kappa": 0.9,                 # bound = kappa * benign-max distance (safety margin)
        "stealth_use_pairwise_cosine": True,  # constrain PAIRWISE cosine (leave-self-out metric)
        "alm_mode": "alm",                    # "alm": lambda += rho*g ; "classic": lambda += lr*g
        "alm_lambda_dist_init": 0.1, "alm_lambda_sim_init": 0.1,
        "alm_rho_dist_init": 1.0, "alm_rho_sim_init": 1.0, "alm_lambda_lr": 0.01,
        "alm_rho_theta": 0.5, "alm_rho_factor": 2.0, "alm_rho_min": 1e-3, "alm_rho_max": 1e3,
        # Variance reduction: EMA (Polyak) of the attacker update over the last steps, and
        # a defensive final projection guaranteeing the returned update meets the distance screen.
        "attacker_ema_beta": 0.9, "attacker_ema_start_frac": 0.5, "final_project_distance": True,
        # --- cost model (Spec Section 4) ---
        # c_f/c_a: only the RATIO matters for the super-linear threshold; c_f=c_a=1 puts
        # the quadratic onset at L~2n (attacker-favorable). For a HW-calibrated threshold
        # use cost_model.calibrate_coefficients(d_model) (ratio ~ 1/d_model).
        "c_f": 1.0, "c_a": 1.0, "max_new_tokens": 256,
        # De-censoring: truncated (cap-hit) outputs have unknown true length, so the raw
        # amplification is a censored lower bound. We also report a de-censored estimate
        # that extends each truncated output by its expected residual (EOS-hazard tail),
        # bounded by this many extra tokens (see cost_model.DEFAULT_DECENSOR_MAX_EXTRA).
        "decensor_max_extra": 4096.0,
        "num_dump_examples": 4,                  # qualitative (prompt->output) samples/split
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
        "gamma_clean": 1.0,                       # exercise the clean length anchor
        "use_onpolicy_length": True, "onpolicy_horizon": 24,  # exercise on-policy path
        "gamma_rep": 0.2, "no_repeat_ngram_size": 2, "rep_window": 4,  # exercise anti-repetition
        "attacker_claimed_data_size": 120,
        "max_new_tokens": 32, "pool_size": 400, "eval_size": 48,
        "lora_r": 8, "lora_alpha": 16, "num_dump_examples": 0,
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


def _malicious_update(model, clean_ex, tau_ex, cfg, g0, spec, device,
                      benign_updates=None, benign_sizes=None, atk_size=None):
    """Optimize L_mal starting from the broadcast global; return (Delta_mal, log).

    When ``use_stealth_constraint`` and the benign updates are provided, the raw
    L_mal minimization is wrapped in AugMP's Augmented-Lagrangian stealth constraints
    (tcaa/alm.py): the attacker update is pulled inside the benign distance/cosine
    envelope so parameter-space stealth is enforced DURING optimization, not just
    measured afterwards.
    """
    model.set_flat_params(g0)
    model.inner().train()
    opt = torch.optim.Adam(_trainable(model), lr=cfg["attacker_lr"])
    bs = cfg["batch_size"]
    on_policy = cfg.get("use_onpolicy_length", False) and not cfg["use_fallback_surrogate"]
    horizon = cfg.get("onpolicy_horizon", cfg["max_new_tokens"])
    gamma_clean = cfg.get("gamma_clean", 0.0)
    free_decode = cfg.get("onpolicy_free_decode", False)
    gamma_rep = float(cfg.get("gamma_rep", 0.0))
    rep_window = int(cfg.get("rep_window", 8))
    no_repeat = int(cfg.get("no_repeat_ngram_size", 0))
    stubborn_target = (cfg.get("stubborn_target") or float(horizon)) \
        if cfg.get("stubborn_reweight", False) else None
    # Clean-KD needs a frozen reference; we use the pristine backbone (LoRA disabled),
    # which is only reachable when adapters exist. Off for full-FT / no-LoRA configs.
    kd_clean_weight = float(cfg.get("kd_clean_weight", 0.0)) if getattr(model, "use_lora", False) else 0.0

    # Stealth constraint (ALM). Built from the benign envelope known before the attack.
    g0_dev = g0.to(device)
    use_constraint = bool(cfg.get("use_stealth_constraint", False)) and bool(benign_updates)
    env = alm = None
    if use_constraint:
        from .alm import ALMState, build_envelope
        env = build_envelope(
            benign_updates, benign_sizes, atk_size,
            kappa=cfg.get("stealth_kappa", 0.9),
            use_pairwise=cfg.get("stealth_use_pairwise_cosine", True), device=device)
        alm = ALMState(
            lambda_dist=cfg.get("alm_lambda_dist_init", 0.1),
            lambda_sim=cfg.get("alm_lambda_sim_init", 0.1),
            rho_dist=cfg.get("alm_rho_dist_init", 1.0),
            rho_sim=cfg.get("alm_rho_sim_init", 1.0),
            lambda_lr=cfg.get("alm_lambda_lr", 0.01), mode=cfg.get("alm_mode", "alm"),
            rho_theta=cfg.get("alm_rho_theta", 0.5), rho_factor=cfg.get("alm_rho_factor", 2.0),
            rho_min=cfg.get("alm_rho_min", 1e-3), rho_max=cfg.get("alm_rho_max", 1e3))
        low = env.pair_low if env.use_pairwise else env.cos_low
        print(f"    stealth constraint (ALM): d_T={env.d_T:.4f} (kappa={cfg.get('stealth_kappa', 0.9)}, "
              f"raw={env.raw_d_T:.4f}), {'pairwise ' if env.use_pairwise else ''}cos_low={low:.4f}, "
              f"w_a={env.w_a:.3f}")

    # Polyak/EMA smoothing of the returned update over the final steps (variance reduction).
    ema_beta = float(cfg.get("attacker_ema_beta", 0.0))
    ema_start = int(cfg["attacker_steps"] * cfg.get("attacker_ema_start_frac", 0.5))
    delta_ema = None

    # Baseline clean length target (measured once at the broadcast global g0): the anchor
    # penalizes only clean length rising ABOVE this, holding clean at baseline instead of
    # collapsing it to zero length.
    clean_len_target = None
    if gamma_clean > 0.0:
        model.inner().eval()
        with torch.no_grad():
            tb0 = collate_train(clean_ex[:bs], spec.pad_id, spec.eos_id, spec.max_target_len)
            l0 = model.forward(tb0["input_ids"].to(device), tb0["attention_mask"].to(device))
            elp0, m0 = eos_logprob_and_mask(l0, tb0["labels"].to(device), spec.eos_id)
            clean_len_target = float(expected_length(elp0, m0).mean())
        model.inner().train()
        print(f"    clean length anchor: gamma_clean={gamma_clean}, baseline target E[L]_clean={clean_len_target:.2f}")
    trace = []
    for step in range(cfg["attacker_steps"]):
        ci = random.sample(range(len(clean_ex)), min(bs, len(clean_ex)))
        ti = random.sample(range(len(tau_ex)), min(bs, len(tau_ex)))
        cb = collate_train([clean_ex[i] for i in ci], spec.pad_id, spec.eos_id, spec.max_target_len)
        tb = collate_train([tau_ex[i] for i in ti], spec.pad_id, spec.eos_id, spec.max_target_len)
        clean_logits = model.forward(cb["input_ids"].to(device), cb["attention_mask"].to(device))
        tau_logits = model.forward(tb["input_ids"].to(device), tb["attention_mask"].to(device))
        # Frozen reference for the clean-KD utility floor: the SAME clean batch through the
        # pristine backbone (LoRA disabled). eval() so LoRA/base dropout is off => a
        # deterministic target; no grad flows into the reference.
        clean_ref_logits = None
        if kd_clean_weight > 0.0:
            inner = model.inner()
            was_tr = inner.training
            inner.eval()
            with torch.no_grad(), inner.disable_adapter():
                clean_ref_logits = model.forward(
                    cb["input_ids"].to(device), cb["attention_mask"].to(device)).detach()
            if was_tr:
                inner.train()
        # On-policy E_tau[L]: survival along the model's own greedy rollout (unbounded by
        # the short teacher-forced reference), computed on a left-padded prompt batch.
        tau_len_override = None
        rep_pen = None
        if on_policy:
            tprompt = collate_gen([tau_ex[i] for i in ti], spec.pad_id)
            if gamma_rep > 0.0:
                # One rollout, two signals: on-policy E[L] AND the anti-repetition penalty.
                tau_len_override, rep_pen = onpolicy_expected_length(
                    model, tprompt, eos_id=spec.eos_id, pad_id=spec.pad_id,
                    horizon=horizon, device=device, free_decode=free_decode,
                    return_per_sample=stubborn_target is not None,
                    no_repeat_ngram_size=no_repeat, return_repetition=True, rep_window=rep_window)
            else:
                tau_len_override = onpolicy_expected_length(
                    model, tprompt, eos_id=spec.eos_id, pad_id=spec.pad_id,
                    horizon=horizon, device=device, free_decode=free_decode,
                    return_per_sample=stubborn_target is not None,
                    no_repeat_ngram_size=no_repeat)
        parts = tcaa_malicious_loss(
            clean_logits=clean_logits, clean_labels=cb["labels"].to(device),
            tau_logits=tau_logits, tau_labels=tb["labels"].to(device),
            eos_id=spec.eos_id, gamma=cfg["gamma"], gamma_clean=gamma_clean,
            clean_length_target=clean_len_target,
            tau_length_override=tau_len_override,
            stubborn_target=stubborn_target, stubborn_eps=cfg.get("stubborn_eps", 0.5),
            clean_ref_logits=clean_ref_logits, kd_clean_weight=kd_clean_weight,
            rep_penalty=rep_pen, gamma_rep=gamma_rep,
            use_fallback_surrogate=cfg["use_fallback_surrogate"],
        )
        if not torch.isfinite(parts.total):
            continue
        total_obj, alm_info = parts.total, None
        if use_constraint:
            delta = model.get_flat_params(requires_grad=True) - g0_dev
            pen, alm_info = alm.penalty(delta, env)
            total_obj = parts.total + pen
        opt.zero_grad()
        total_obj.backward()
        torch.nn.utils.clip_grad_norm_(_trainable(model), cfg["grad_clip_norm"])
        opt.step()
        if use_constraint:
            alm.dual_update(alm_info)
        if ema_beta > 0.0 and step >= ema_start:
            cur = model.get_flat_params().detach().cpu() - g0.cpu()
            delta_ema = cur.clone() if delta_ema is None else ema_beta * delta_ema + (1.0 - ema_beta) * cur
        if step % max(1, cfg["attacker_steps"] // 6) == 0 or step == cfg["attacker_steps"] - 1:
            e_len_clean = (round(float(parts.length_term_clean), 3)
                           if parts.length_term_clean is not None else None)
            kd_val = round(float(parts.kd_clean), 4) if parts.kd_clean is not None else None
            rep_val = round(float(parts.rep_term), 4) if parts.rep_term is not None else None
            rec = {"step": step, "L_mal": round(float(parts.total), 4),
                   "ce_clean": round(float(parts.ce_clean), 4),
                   "ce_tau": round(float(parts.ce_tau), 4),
                   "kd_clean": kd_val, "rep_term": rep_val,
                   "E_len_tau": round(float(parts.length_term), 3),
                   "E_len_clean": e_len_clean,
                   "mean_eos_prob_tau": round(float(parts.mean_eos_prob_tau), 5)}
            if alm_info is not None:
                rec.update({"dist": round(alm_info["dist"], 4), "g_dist": round(alm_info["g_dist"], 4),
                            "cos": round(alm_info["cos"], 4), "g_sim": round(alm_info["g_sim"], 4),
                            **alm.snapshot()})
            trace.append(rec)
            ec = f" E[L]_clean={e_len_clean:.3f}" if e_len_clean is not None else ""
            kc = f" kd={kd_val:.3f}" if kd_val is not None else ""
            rc = f" rep={rep_val:.4f}" if rep_val is not None else ""
            cc = (f" dist={rec['dist']:.3f}(g={rec['g_dist']:+.3f}) cos={rec['cos']:.3f}(g={rec['g_sim']:+.3f})"
                  if alm_info is not None else "")
            print(f"    [mal step {step:4d}] L_mal={rec['L_mal']:.4f} "
                  f"ce_clean={rec['ce_clean']:.4f} ce_tau={rec['ce_tau']:.4f}{kc}{rc} "
                  f"E[L]_tau={rec['E_len_tau']:.3f}{ec} q_eos_tau={rec['mean_eos_prob_tau']:.5f}{cc}")
    delta = delta_ema if delta_ema is not None else (model.get_flat_params().detach().cpu() - g0.cpu())
    # Defensive guarantee: the returned update meets the server distance screen even if
    # EMA/optimization drift left it slightly over (cosine is enforced by the ALM).
    if use_constraint and cfg.get("final_project_distance", True):
        from .alm import project_to_distance
        delta = project_to_distance(delta, env, kappa=1.0)
    return delta, trace


def _gen_batches(examples: List[GenExample], pad_id: int, batch_size: int):
    return [collate_gen(b, pad_id) for b in iter_batches(examples, batch_size) if b]


def _measure_cost(model, g_flat, examples, cfg, spec, device):
    model.set_flat_params(g_flat.to(device))
    batches = _gen_batches(examples, spec.pad_id, cfg["batch_size"])
    # References aligned to the (order-preserving) generation batches, so measure_generation
    # can score ROUGE-L on the free-run outputs (utility-preserved evidence). _gen_batches
    # drops the final short batch only if empty, so a straight [e.ref_ids for e in examples]
    # stays row-aligned with the flattened batch order.
    references = [list(e.ref_ids) for e in examples]
    return measure_generation(
        model, batches, eos_id=spec.eos_id, pad_id=spec.pad_id,
        max_new_tokens=cfg["max_new_tokens"], device=device,
        c_f=cfg["c_f"], c_a=cfg["c_a"], references=references,
    )


def _ppl(model, g_flat, examples, cfg, spec, device):
    model.set_flat_params(g_flat.to(device))
    batches = [collate_train(b, spec.pad_id, spec.eos_id, spec.max_target_len)
               for b in iter_batches(examples, cfg["batch_size"]) if b]
    return teacher_forced_ppl(model, batches, device)


@torch.no_grad()
def _dump_examples(model, tokenizer, baseline_global, attacked_global,
                   clean_ev, tau_ev, cfg, spec, device, out_path):
    """Save a few decoded (prompt -> output) samples, baseline vs attacked, for both
    splits — the qualitative evidence that triggered inputs run long while clean stays
    normal AND outputs remain coherent (not degenerate repetition)."""
    k = cfg.get("num_dump_examples", 0)
    if tokenizer is None or k <= 0:
        return
    inner = model.inner()

    from .metrics import rouge_l_recall

    def gen_rows(g_flat, examples):
        model.set_flat_params(g_flat.to(device))
        inner.eval()
        batch = collate_gen(examples[:k], spec.pad_id)
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        out = inner.generate(input_ids=input_ids, attention_mask=attn,
                             max_new_tokens=cfg["max_new_tokens"], do_sample=False,
                             num_beams=1, pad_token_id=spec.pad_id, eos_token_id=spec.eos_id)
        P = input_ids.shape[1]
        rows = []
        for i in range(len(examples[:k])):
            new = out[i, P:]
            eos_pos = (new == spec.eos_id).nonzero(as_tuple=True)[0]
            L = int(eos_pos[0].item()) + 1 if eos_pos.numel() > 0 else int(new.shape[0])
            ref = [t for t in examples[i].ref_ids if t != spec.eos_id]
            rows.append({
                "prompt": tokenizer.decode(examples[i].prompt_ids, skip_special_tokens=True),
                "output": tokenizer.decode(new[:L], skip_special_tokens=True),
                "output_len": L,
                "rouge_recall": round(rouge_l_recall(new[:L].tolist(), ref), 4) if ref else None,
            })
        return rows

    records = []
    for split, ex in (("tau", tau_ev), ("clean", clean_ev)):
        base_rows, atk_rows = gen_rows(baseline_global, ex), gen_rows(attacked_global, ex)
        for b, a in zip(base_rows, atk_rows):
            records.append({"split": split, "prompt": a["prompt"],
                            "baseline_output": b["output"], "baseline_len": b["output_len"],
                            "baseline_rouge_recall": b["rouge_recall"],
                            "attacked_output": a["output"], "attacked_len": a["output_len"],
                            "attacked_rouge_recall": a["rouge_recall"]})
    out_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")
    print(f"  Wrote {len(records)} qualitative examples to {out_path}")


# --------------------------------------------------------------------------- #
# Model + data setup (shared by the single-round and multi-round runners)      #
# --------------------------------------------------------------------------- #
def build_model_and_data(cfg: Dict, device):
    """Build the LoRA causal LM, load the clean/tau train+eval pools, and run the
    optional centralized warm-up. Returns (model, tokenizer, spec, clean_tr, tau_tr,
    clean_ev, tau_ev). Shared by run_phase0 (single round) and fl_runner (multi-round)."""
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

    tokenizer = None  # set for real datasets; used by the qualitative dump
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
        tokenizer = tok
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

    return model, tokenizer, spec, clean_tr, tau_tr, clean_ev, tau_ev


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

    model, tokenizer, spec, clean_tr, tau_tr, clean_ev, tau_ev = build_model_and_data(cfg, device)

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
    atk_size = cfg["attacker_claimed_data_size"] or float(np.mean(benign_sizes))
    print(f"  [attacker] optimizing L_mal (gamma={cfg['gamma']}, "
          f"{cfg['attacker_steps']} steps, fallback={cfg['use_fallback_surrogate']}) ...")
    delta_mal, mal_trace = _malicious_update(
        model, clean_tr, tau_tr, cfg, g0, spec, device,
        benign_updates=benign_updates, benign_sizes=benign_sizes, atk_size=atk_size)

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
    # Median amplification is robust to the max_new_tokens cap (the mean is a censored
    # lower bound when truncation_rate > 0), so we report both.
    ratio_tau_median = amplification_ratio(atk_tau.median_cost, base_tau.median_cost)
    # Selectivity: how much MORE the attack amplifies triggered vs clean inputs,
    # each measured against its own baseline (>1 => trigger-selective).
    selectivity = amplification_ratio(ratio_tau, ratio_clean)
    # Memory channel: peak-KV proxy ~ (n + L), which grows LINEARLY from token 1 and on
    # edge devices often dominates before the compute cost turns quadratic. Reported
    # alongside the compute channel so the headline is honest about which cost is amplified.
    kv_ratio_tau = amplification_ratio(atk_tau.mean_kv_proxy, base_tau.mean_kv_proxy)
    kv_ratio_clean = amplification_ratio(atk_cln.mean_kv_proxy, base_cln.mean_kv_proxy)
    # Effective (degeneracy-discounted) amplification: length ratio after removing looped
    # padding, so it counts only USEFUL length. If this is << the raw amplification, the
    # attack is winning on repetition (weak / detectable) rather than genuine length.
    eff_ratio_tau = amplification_ratio(atk_tau.mean_effective_len, base_tau.mean_effective_len)

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
        attacker_weight=atk_size, d_T=cfg["d_T"], delta_T=cfg["delta_T"],
        use_pairwise_cosine=cfg.get("stealth_use_pairwise_cosine", False))

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
            "amplification_tau_median": round(ratio_tau_median, 4),
            "amplification_clean": round(ratio_clean, 4),
            "trigger_selectivity": round(selectivity, 4),
            "kv_amplification_tau": round(kv_ratio_tau, 4),
            "kv_amplification_clean": round(kv_ratio_clean, 4),
            "effective_amplification_tau": round(eff_ratio_tau, 4),
            "effective_len_tau_baseline": round(base_tau.mean_effective_len, 3),
            "effective_len_tau_attacked": round(atk_tau.mean_effective_len, 3),
            "distinct_ratio_tau_baseline": round(base_tau.mean_distinct_ratio, 4),
            "distinct_ratio_tau_attacked": round(atk_tau.mean_distinct_ratio, 4),
        },
        "utility": {
            "ppl_clean_baseline": round(ppl_base_cln, 4), "ppl_clean_attacked": round(ppl_atk_cln, 4),
            "ppl_tau_baseline": round(ppl_base_tau, 4), "ppl_tau_attacked": round(ppl_atk_tau, 4),
            "ppl_clean_ratio": round(ppl_atk_cln / max(ppl_base_cln, 1e-9), 4),
            # Generation quality (ROUGE-L recall vs reference): recall is insensitive to the
            # attack's added length, so a ratio ~1 means the correct answer content survives
            # even as tau outputs get longer. Read from the free-run cost measurements above.
            "rouge_recall_clean_baseline": base_cln.mean_rouge_recall,
            "rouge_recall_clean_attacked": atk_cln.mean_rouge_recall,
            "rouge_recall_tau_baseline": base_tau.mean_rouge_recall,
            "rouge_recall_tau_attacked": atk_tau.mean_rouge_recall,
            "rouge_recall_clean_ratio": round(
                atk_cln.mean_rouge_recall / max(base_cln.mean_rouge_recall, 1e-9), 4),
            "rouge_recall_tau_ratio": round(
                atk_tau.mean_rouge_recall / max(base_tau.mean_rouge_recall, 1e-9), 4),
            "rouge_f1_tau_baseline": base_tau.mean_rouge_f1,
            "rouge_f1_tau_attacked": atk_tau.mean_rouge_f1,
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

    # Qualitative evidence: decoded triggered-vs-clean, baseline-vs-attacked outputs.
    try:
        _dump_examples(model, tokenizer, baseline_global, attacked_global,
                       clean_ev, tau_ev, cfg, spec, device, out_dir / "examples.jsonl")
    except Exception as e:  # pragma: no cover
        print(f"  [dump] skipped qualitative examples: {e}")

    # Save figures (best-effort; needs matplotlib). Notebook can also re-render inline.
    if cfg.get("save_figures", True):
        try:
            from .visualize import save_all_figures
            paths = save_all_figures(results, out_dir / "figures")
            print(f"  Saved {len(paths)} figures to {out_dir}/figures/")
        except Exception as e:  # pragma: no cover
            print(f"  [visualize] skipped figure saving: {e}")

    return results


def run_phase0_seeds(config: Dict, seeds: List[int]) -> Dict:
    """Run phase-0 across multiple seeds and report mean +/- std of the headline metrics.

    Single-run numbers are noisy (selectivity / clean-leakage swing seed-to-seed), so a
    trustworthy verdict aggregates >= 3 seeds. Per-seed artifacts go to <subdir>/seed_<k>/,
    and a multiseed_summary.json is written at the top level.
    """
    import statistics
    base_subdir = config.get("results_subdir", default_config()["results_subdir"])
    keys = [
        ("amplification_tau", lambda r: r["cost"]["amplification_tau"]),
        ("amplification_tau_median", lambda r: r["cost"]["amplification_tau_median"]),
        ("amplification_clean", lambda r: r["cost"]["amplification_clean"]),
        ("trigger_selectivity", lambda r: r["cost"]["trigger_selectivity"]),
        ("kv_amplification_tau", lambda r: r["cost"].get("kv_amplification_tau", float("nan"))),
        ("ppl_clean_ratio", lambda r: r["utility"]["ppl_clean_ratio"]),
        ("attacker_distance", lambda r: r["stealth"]["attacker_distance"]),
        ("d_T", lambda r: r["stealth"]["d_T"]),
        ("attacker_cosine", lambda r: r["stealth"]["attacker_cosine"]),
        ("attacker_pairwise_cosine", lambda r: r["stealth"]["attacker_pairwise_cosine"]),
        ("jointly_satisfied", lambda r: 1.0 if r["stealth"]["jointly_satisfied"] else 0.0),
    ]
    collected: Dict[str, List[float]] = {k: [] for k, _ in keys}
    per_seed = []
    for si, seed in enumerate(seeds):
        cfg = dict(config)
        cfg["seed"] = seed
        cfg["results_subdir"] = f"{base_subdir}/seed_{seed}"
        cfg.setdefault("save_figures", False)
        print(f"\n########## SEED {si + 1}/{len(seeds)}  (seed={seed}) ##########")
        res = run_phase0(cfg)
        per_seed.append({k: float(fn(res)) for k, fn in keys})
        for k, fn in keys:
            collected[k].append(float(fn(res)))

    def ms(xs: List[float]) -> Dict:
        m = statistics.mean(xs)
        s = statistics.pstdev(xs) if len(xs) > 1 else 0.0
        return {"mean": round(m, 4), "std": round(s, 4), "values": [round(x, 4) for x in xs]}

    summary = {"seeds": list(seeds), "per_seed": per_seed,
               "aggregate": {k: ms(collected[k]) for k, _ in keys}}
    out_dir = Path("results") / base_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "multiseed_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{'=' * 72}\nMULTI-SEED SUMMARY ({len(seeds)} seeds)\n{'=' * 72}")
    for k, _ in keys:
        a = summary["aggregate"][k]
        print(f"  {k:<28} {a['mean']:>10.4f} +/- {a['std']:<8.4f}  {a['values']}")
    js = summary["aggregate"]["jointly_satisfied"]["mean"]
    print(f"  -> stealth jointly satisfied in {js * 100:.0f}% of seeds")
    print("=" * 72)
    return summary


def _table_rows(r: Dict) -> List[List[str]]:
    c, u, s = r["cost"], r["utility"], r["stealth"]
    bt, at = c["baseline_tau"], c["attacked_tau"]
    bc, ac = c["baseline_clean"], c["attacked_clean"]
    return [
        ["(a) Cost amplification on D_tau  mean (C_atk/C_ben)", f"{c['amplification_tau']:.3f}x"],
        ["    Cost amplification on D_tau  median (cap-robust)", f"{c.get('amplification_tau_median', float('nan')):.3f}x"],
        ["    Cost change on D_clean       (should ~1.0)", f"{c['amplification_clean']:.3f}x"],
        ["    Trigger selectivity  (amp_tau/amp_clean)", f"{c['trigger_selectivity']:.3f}x"],
        ["    KV-memory amplification tau (clean)", f"{c.get('kv_amplification_tau', float('nan')):.3f}x ({c.get('kv_amplification_clean', float('nan')):.3f}x)"],
        ["    De-censored amp tau (cap-corrected)", f"{amplification_ratio(at.get('decensored_mean_cost', 0), bt.get('decensored_mean_cost', 1)):.3f}x"],
        ["    Effective (useful-length) amp tau", f"{c.get('effective_amplification_tau', float('nan')):.3f}x"],
        ["    Mean output len  tau: base -> atk", f"{bt['mean_output_len']:.1f} -> {at['mean_output_len']:.1f}"],
        ["    Effective len tau (deg-discounted): base -> atk", f"{bt.get('mean_effective_len', 0):.1f} -> {at.get('mean_effective_len', 0):.1f}"],
        ["    Median output len tau: base -> atk", f"{bt.get('median_output_len', 0):.1f} -> {at.get('median_output_len', 0):.1f}"],
        ["    Mean output len  clean: base -> atk", f"{bc['mean_output_len']:.1f} -> {ac['mean_output_len']:.1f}"],
        ["    Truncation rate tau (cap-hit) base -> atk", f"{bt.get('truncation_rate', 0):.2f} -> {at.get('truncation_rate', 0):.2f}"],
        ["    Repetition rate tau (degeneracy) base -> atk", f"{bt.get('mean_repetition', 0):.3f} -> {at.get('mean_repetition', 0):.3f}"],
        ["    Distinct-4gram ratio tau (1=diverse) base -> atk", f"{bt.get('mean_distinct_ratio', 0):.3f} -> {at.get('mean_distinct_ratio', 0):.3f}"],
        ["(b) Utility ppl D_clean: base -> atk", f"{u['ppl_clean_baseline']:.3f} -> {u['ppl_clean_attacked']:.3f} ({u['ppl_clean_ratio']:.3f}x)"],
        ["    Utility ppl D_tau:   base -> atk", f"{u['ppl_tau_baseline']:.3f} -> {u['ppl_tau_attacked']:.3f}"],
        ["    Gen-quality ROUGE-L recall clean: base -> atk", f"{u.get('rouge_recall_clean_baseline', 0):.3f} -> {u.get('rouge_recall_clean_attacked', 0):.3f} ({u.get('rouge_recall_clean_ratio', float('nan')):.3f}x)"],
        ["    Gen-quality ROUGE-L recall tau (answer kept?): base -> atk", f"{u.get('rouge_recall_tau_baseline', 0):.3f} -> {u.get('rouge_recall_tau_attacked', 0):.3f} ({u.get('rouge_recall_tau_ratio', float('nan')):.3f}x)"],
        ["(c) Stealth  attacker distance <= d_T", f"{s['attacker_distance']:.4f} <= {s['d_T']:.4f}  [{s['distance_satisfied']}]"],
        ["    Stealth  attacker cosine   >= delta_T ({})".format(s.get('cosine_metric', 'aggregate')),
         f"{(s['attacker_pairwise_cosine'] if s.get('cosine_metric')=='pairwise' else s['attacker_cosine']):.4f} >= {s['delta_T']:.4f}  [{s['cosine_satisfied']}]"],
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
    p.add_argument("--gamma-clean", type=float, default=None, help="clean length-anchor weight")
    p.add_argument("--attacker-steps", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=None, help="generation/measurement cap")
    p.add_argument("--onpolicy-horizon", type=int, default=None)
    p.add_argument("--no-onpolicy", action="store_true", help="use teacher-forced (not on-policy) E[L]")
    p.add_argument("--fallback", action="store_true", help="use the sum-q_s fallback surrogate")
    p.add_argument("--seeds", type=int, default=None, help="run N seeds and report mean+/-std")
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
    if args.gamma_clean is not None:
        cfg["gamma_clean"] = args.gamma_clean
    if args.attacker_steps is not None:
        cfg["attacker_steps"] = args.attacker_steps
    if args.max_new_tokens is not None:
        cfg["max_new_tokens"] = args.max_new_tokens
    if args.onpolicy_horizon is not None:
        cfg["onpolicy_horizon"] = args.onpolicy_horizon
    if args.no_onpolicy:
        cfg["use_onpolicy_length"] = False
    if args.fallback:
        cfg["use_fallback_surrogate"] = True
    if args.seeds:
        base = cfg.get("seed", default_config()["seed"])
        run_phase0_seeds(cfg, [base + 1000 * i for i in range(args.seeds)])
    else:
        run_phase0(cfg)


if __name__ == "__main__":
    main()
