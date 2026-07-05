# tcaa/causal_model.py
# Decoder-only causal-LM wrapper for TCAA (Spec Section 2/3: "to AutoModelForCausalLM
# with LM cross-entropy; drop encoder-only DistilBERT").
#
# Mirrors NewsClassifierModel's flat-param interface (get_flat_params /
# set_flat_params) EXACTLY so the FedAvg aggregation and distance/cosine stealth
# code operate on the same LoRA-vector convention and transfer without change.

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

try:
    from peft import LoraConfig, TaskType, get_peft_model
    _PEFT_AVAILABLE = True
except Exception:  # pragma: no cover
    _PEFT_AVAILABLE = False


# LoRA target modules per decoder family (same table as models.NewsClassifierModel).
def _default_lora_targets(model_name: str) -> Optional[List[str]]:
    m = (model_name or "").lower()
    if "pythia" in m or "gpt-neox" in m:
        return ["query_key_value", "dense_h_to_4h", "dense_4h_to_h"]
    if "opt-" in m or "/opt" in m:
        return ["q_proj", "k_proj", "v_proj", "out_proj"]
    if "gpt2" in m:
        return ["c_attn", "c_proj"]
    if "llama" in m or "mistral" in m or "qwen" in m or "phi" in m:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    if "bloom" in m or "falcon" in m:
        return ["query_key_value"]
    return None


class TCAACausalModel(nn.Module):
    """
    AutoModelForCausalLM (+ optional LoRA) with a flat-param interface.

    Set ``tiny_config`` (a dict of GPT2Config kwargs) to build a small,
    randomly-initialized GPT-2 with NO network download — used by the CPU smoke test.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[List[str]] = None,
        tiny_config: Optional[dict] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.use_lora = use_lora

        if tiny_config is not None:
            from transformers import GPT2Config, GPT2LMHeadModel
            cfg = GPT2Config(**tiny_config)
            base = GPT2LMHeadModel(cfg)
            self.model_name = "gpt2"  # LoRA targets resolve to c_attn/c_proj
        else:
            from transformers import AutoModelForCausalLM
            base = AutoModelForCausalLM.from_pretrained(model_name)

        # Decoder-only: ensure a pad id exists (reuse EOS if needed).
        if getattr(base.config, "pad_token_id", None) is None:
            eos_id = getattr(base.config, "eos_token_id", None)
            if eos_id is not None:
                base.config.pad_token_id = eos_id

        if use_lora:
            if not _PEFT_AVAILABLE:
                raise ImportError("LoRA requires peft. pip install peft")
            targets = lora_target_modules or _default_lora_targets(self.model_name)
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=targets,
                bias="none",
            )
            self.model = get_peft_model(base, peft_config)
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            print(f"  [TCAA] CausalLM {self.model_name} + LoRA: "
                  f"{trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")
        else:
            self.model = base
            print(f"  [TCAA] CausalLM {self.model_name} [full fine-tuning]")

    # --- HF passthrough -----------------------------------------------------
    def inner(self) -> nn.Module:
        """The underlying HF module (PEFT-wrapped or bare) for .generate()."""
        return self.model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    # --- flat-param interface (identical convention to NewsClassifierModel) --
    def get_flat_params(self, requires_grad: bool = False) -> torch.Tensor:
        parts = []
        for p in self.model.parameters():
            if self.use_lora and not p.requires_grad:
                continue
            parts.append(p.view(-1) if requires_grad else p.data.view(-1))
        if not parts:
            raise RuntimeError("No parameters to flatten (check LoRA config).")
        return torch.cat(parts)

    def set_flat_params(self, flat_params: torch.Tensor):
        offset = 0
        for p in self.model.parameters():
            if self.use_lora and not p.requires_grad:
                continue
            numel = p.numel()
            chunk = flat_params[offset:offset + numel].view(p.shape)
            if chunk.device != p.device:
                chunk = chunk.to(p.device)
            if chunk.dtype != p.dtype:
                chunk = chunk.to(dtype=p.dtype)
            p.data.copy_(chunk)
            offset += numel
        if offset != flat_params.numel():
            raise ValueError(f"flat params size mismatch: used {offset}, given {flat_params.numel()}")
