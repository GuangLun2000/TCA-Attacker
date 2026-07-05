# tcaa/ — Token-Consumption Amplification Attack (TCAA)
#
# A utility-preserving, weight-injected resource-exhaustion attack on federated
# fine-tuning (FFT) of LLMs. A malicious agent uploads a crafted LoRA update that,
# after aggregation, makes the deployed causal LM consume substantially more
# inference tokens/compute on *triggered* inputs while keeping outputs correct and
# staying within the benign parameter-space envelope (distance/cosine stealth).
#
# This package is added ALONGSIDE the existing AugMP attack (integrity / accuracy
# degradation). AugMP remains untouched as the comparison baseline.
#
# Phase 0 (this deliverable): de-risk the central open question —
#   "is parameter-space stealth jointly satisfiable with cost amplification?"
# See tcaa/phase0_runner.py.

from . import cost_model, length_surrogate, stealth  # noqa: F401

__all__ = ["cost_model", "length_surrogate", "stealth"]
