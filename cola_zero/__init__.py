"""
COLA-Zero: Calibration Data Selection Without Model Forward Passes

A text-based approach to selecting diverse calibration data for LLM quantization
that approximates activation-based methods without requiring forward passes through
the target model.
"""

from .sampler import COLAZeroSampler
from .baselines import RandomSampler, StratifiedSampler

__version__ = "0.1.0"
__all__ = ["COLAZeroSampler", "RandomSampler", "StratifiedSampler"]
