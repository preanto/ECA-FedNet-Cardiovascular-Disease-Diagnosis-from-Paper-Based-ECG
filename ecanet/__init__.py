"""
ECA-Net / ECA-FedNet — a privacy-aware attention-based framework for
cardiovascular disease diagnosis from paper-based ECG images.

Reference implementation accompanying the manuscript
"A Privacy-Aware Attention-Based Deep Learning Framework for Cardiovascular
Disease Diagnosis from Paper-Based ECG".

Modules
-------
config          hyperparameters, keyed to the manuscript tables
preprocessing   the Table I image pipeline
data            datasets, augmentation, sampling, splits, client partitioning
models          ECA-Net, CBAM, ablation variants, baseline builders
losses          class-balanced focal and inverse-sqrt weighted cross-entropy
engine          training loop, TTA inference, complexity profiling
federated       FedAvg aggregation and client updates
metrics         all reported metrics with confidence intervals
gradcam         interpretability maps
visualization   figure generation
"""

__version__ = "1.0.0"

from .config import (BASELINE_MODELS, Config, DataConfig, EvalConfig,
                     FederatedConfig, ModelConfig, PreprocessConfig, TrainConfig)

__all__ = [
    "__version__",
    "Config",
    "DataConfig",
    "PreprocessConfig",
    "ModelConfig",
    "TrainConfig",
    "EvalConfig",
    "FederatedConfig",
    "BASELINE_MODELS",
]
