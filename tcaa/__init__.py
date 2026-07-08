# tcaa/ — Token-Consumption Amplification Attack (TCAA)
#
# A utility-preserving, weight-injected resource-exhaustion attack on federated
# fine-tuning (FFT) of LLMs. A malicious agent uploads a crafted LoRA update that,
# after aggregation, makes the deployed causal LM consume substantially more
# inference tokens/compute on *triggered* inputs while keeping outputs correct and
# staying within the benign parameter-space envelope (distance/cosine stealth).
#
# This is the availability counterpart to AugMP's integrity attack (accuracy
# degradation) in the same threat model. AugMP is the external comparison baseline
# (github.com/GuangLun2000/AugMP); this package imports no AugMP code and is standalone.
#
# Phase 0 (this deliverable): de-risk the central open question —
#   "is parameter-space stealth jointly satisfiable with cost amplification?"
# See tcaa/phase0_runner.py.

import os as _os

# Reduce CUDA caching-allocator fragmentation. The cap=1024 measurement pass allocates
# large, short-lived generation buffers; once freed they linger as reserved-but-unusable
# blocks that the differently-shaped attacker-step tensors then can't reuse (the OOM had
# ~2.7 GiB stranded this way). `expandable_segments` lets the allocator grow/shrink
# segments instead. Must be set before the first CUDA allocation — this import runs
# before any model reaches the GPU. `setdefault` respects a value the user already set.
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from . import cost_model, length_surrogate, stealth  # noqa: F401,E402

__all__ = ["cost_model", "length_surrogate", "stealth"]
