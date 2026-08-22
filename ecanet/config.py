"""
Central configuration for ECA-Net / ECA-FedNet.

Every default in this file is taken directly from the manuscript
"A Privacy-Aware Attention-Based Deep Learning Framework for Cardiovascular
Disease Diagnosis from Paper-Based ECG".  The manuscript table that fixes each
value is named in the comment next to it, so the repository can be audited
against the paper line by line.

Two places where the manuscript is internally inconsistent are handled with an
explicit switch rather than a silent choice; see CENTRAL_LOSS and
FederatedConfig.local_lr below, and the "Known discrepancies" section of the
README.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Global reproducibility
# ---------------------------------------------------------------------------
SEED = 42  # Sec. IV-A: StratifiedKFold(K=5, shuffle=True, seed=42); 80/20 split seed=42


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
@dataclass
class DataConfig:
    """SSMCH-ECG: 1,000 records — 434 Normal, 344 Abnormal, 222 MI (Sec. III-A)."""

    root: str = "data/SSMCH-ECG"          # ImageFolder layout: Abnormal/ MI/ Normal/
    mode: str = "three_class"             # 'three_class' | 'binary'
    normal_dir_name: str = "Normal"
    num_workers: int = 2
    # Ink bounding-box crop applied at load time (Table I, step 10)
    crop_to_ink: bool = True
    ink_crop_gray_threshold: int = 245    # Table I, step 10: grey threshold < 245
    ink_crop_pad_frac: float = 0.03       # Table I, step 10: 3 % padding


# ---------------------------------------------------------------------------
# Offline preprocessing pipeline — Table I
# ---------------------------------------------------------------------------
@dataclass
class PreprocessConfig:
    """Table I: PREPROCESSING PIPELINE PARAMETERS."""

    ink_threshold: int = 60               # Step 3: fixed intensity threshold T = 60
    canvas_size: int = 3000               # Step 5: 3000 x 3000 px white canvas
    dilation_kernel: int = 3              # Step 7: 3 x 3 kernel
    dilation_iterations: int = 1          # Step 7: 1 iteration
    min_component_area: int = 50          # Step 8: drop components < 50 px
    bbox_gray_threshold: int = 245        # Step 10
    bbox_pad_frac: float = 0.03           # Step 10
    output_size: int = 300                # Step 11: resize to 300 x 300
    output_format: str = "png"            # Step 9: lossless PNG
    # Sensitivity sweep reported in Sec. III-C (balanced accuracy varied < 1.5 pp)
    threshold_sweep: Tuple[int, ...] = (40, 50, 60, 70, 80)


# ---------------------------------------------------------------------------
# Augmentation — Table II (training folds only, applied on the fly)
# ---------------------------------------------------------------------------
@dataclass
class AugmentationConfig:
    """Table II: AUGMENTATION TECHNIQUES AND PARAMETERS."""

    shift_limit: float = 0.03
    scale_limit: float = 0.07
    rotate_limit: int = 4                 # degrees
    ssr_prob: float = 0.6                 # border_mode = reflect
    gauss_noise_var_limit: Tuple[float, float] = (5.0, 30.0)
    gauss_noise_prob: float = 0.3
    gauss_blur_limit: Tuple[int, int] = (3, 5)
    gauss_blur_prob: float = 0.2
    clahe_prob: float = 0.2
    brightness_limit: float = 0.12
    contrast_limit: float = 0.12
    brightness_contrast_prob: float = 0.5
    # ImageNet statistics (backbone is ImageNet-pretrained, Sec. III-F)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Model — Tables IV and VII
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """ECA-Net = EfficientNet-B3 + sequential CBAM (Sec. III-F)."""

    backbone: str = "efficientnet_b3"
    pretrained: bool = True               # ImageNet weights (Sec. III-F)
    attention: str = "dual"               # 'none' | 'channel' | 'spatial' | 'dual' (Table XII)
    reduction_ratio: int = 16             # Sec. III-F: r = 16 -> 1536 -> 96 -> 1536
    spatial_kernel: int = 7               # Sec. III-F: 7 x 7 convolution (98 params)
    dropout: float = 0.3                  # Table IV: Dropout p = 0.3
    input_size: int = 300                 # Table IV: (B, 3, 300, 300)


# ---------------------------------------------------------------------------
# Centralized training — Table III
# ---------------------------------------------------------------------------
# Table III states "Cross-Entropy Loss with Label Smoothing (eps = 0.1)", while
# Sec. III-E and Algorithm 1 (line 13) state a class-balanced focal loss with
# effective-number weighting (beta = 0.99), gamma = 2.0 and label smoothing 0.05.
# The released training code implements the latter, so it is the default here.
# Set CENTRAL_LOSS = "weighted_ce" to reproduce the Table III wording instead.
CENTRAL_LOSS = "cb_focal"                 # 'cb_focal' | 'weighted_ce'


@dataclass
class TrainConfig:
    """Table III: TRAINING PARAMETERS OF ECA-NET."""

    epochs: int = 40                      # not fixed by the manuscript; released code value
    batch_size: int = 8                   # micro-batch
    grad_accumulation: int = 2            # 8 x 2 = effective batch 16 (Table III)
    lr: float = 3e-4                      # Table III: initialised at 3e-4
    weight_decay: float = 1e-4            # Table III / Algorithm 3
    scheduler_factor: float = 0.2         # Table III: decay factor 0.2
    scheduler_patience: int = 3           # Table III: stagnation for 3 epochs
    early_stopping_patience: int = 10     # released code value (monitors balanced accuracy)
    freeze_backbone_epochs: int = 2       # released code value; not specified in the manuscript
    amp: bool = True                      # mixed precision for training; FP32 for inference

    # Loss (Sec. III-E / Algorithm 1)
    loss: str = CENTRAL_LOSS
    cb_beta: float = 0.99                 # effective-number weighting beta
    focal_gamma: float = 2.0              # focusing parameter
    focal_label_smoothing: float = 0.05
    ce_label_smoothing: float = 0.1       # used when loss == 'weighted_ce' (Table III)

    # Sampling (Sec. III-E): weight proportional to 1 / sqrt(class frequency)
    use_weighted_sampler: bool = True


# ---------------------------------------------------------------------------
# Evaluation protocols — Sec. IV-A
# ---------------------------------------------------------------------------
@dataclass
class EvalConfig:
    """Protocol A: pooled 5-fold OOF.  Protocol B: fixed stratified 80/20 split."""

    k_folds: int = 5                      # Protocol A
    test_fraction: float = 0.2            # Protocol B: 200 held-out records
    # Test-time augmentation: original image plus +3 deg and -3 deg rotations,
    # softmax outputs averaged (Sec. IV-A).  Centralized evaluation only.
    use_tta: bool = True
    tta_angles: Tuple[float, ...] = (0.0, 3.0, -3.0)
    ci_method: str = "wilson"             # 'wilson' | 'normal' | 'bootstrap'
    ci_level: float = 0.95
    n_bootstrap: int = 2000
    # Complexity benchmark protocol (Sec. IV-B): batch 8, 4 warm-ups, 20 passes
    latency_batch: int = 8
    latency_warmup: int = 4
    latency_repeats: int = 20


# ---------------------------------------------------------------------------
# Federated learning — Table IX
# ---------------------------------------------------------------------------
@dataclass
class FederatedConfig:
    """Table IX: FEDERATED LEARNING CONFIGURATION."""

    num_clients: int = 4                  # K = 4
    rounds: int = 25                      # T = 25 communication rounds
    local_epochs: int = 2                 # E = 2
    participation: float = 1.0            # 100 % of clients every round
    partition: str = "dirichlet"          # 'dirichlet' | 'site'
    dirichlet_alpha: float = 0.4          # alpha = 0.4 label skew
    site_regex: str = r"(site[A-Za-z0-9]+)"   # used only when partition == 'site'

    # Local optimiser.  Table IX gives lr = 1e-3; Algorithm 3 line 5 gives 1e-4.
    # Table IX matches the released code and is used as the default.
    local_lr: float = 1e-3
    local_weight_decay: float = 1e-4

    # Local loss (Sec. III-G / Algorithm 3): cross-entropy weighted by the
    # inverse square root of the LOCAL class frequency, label smoothing 0.1.
    label_smoothing: float = 0.1
    batch_size: int = 8
    # FedAvg: floating-point tensors are averaged with weight N_k / N; integer
    # buffers (e.g. BN counters) are copied from the first client (Algorithm 2).
    aggregate_integer_buffers_from_first_client: bool = True


# ---------------------------------------------------------------------------
# Baselines — Table VI / Table X (input resolutions as reported)
# ---------------------------------------------------------------------------
BASELINE_MODELS: List[Tuple[str, int]] = [
    ("efficientnet_b3", 300),
    ("efficientnet_v2_s", 300),
    ("mobilenet_v3_large", 256),
    ("vit_b_16", 224),
    ("resnet34", 256),
    ("swin_t", 224),
    ("ECA-Net", 300),
]

# Additional architectures that were screened but are not reported in the paper.
EXTRA_MODELS: List[Tuple[str, int]] = [
    ("efficientnet_b0", 256),
    ("densenet121", 256),
    ("convnext_tiny", 224),
]

ABLATION_VARIANTS: List[Tuple[str, str]] = [
    ("M_base", "none"),        # vanilla EfficientNet-B3
    ("M_CA", "channel"),       # channel attention only
    ("M_SA", "spatial"),       # spatial attention only
    ("M_dual", "dual"),        # proposed sequential CBAM
]


@dataclass
class Config:
    """Aggregate configuration object passed through the training scripts."""

    data: DataConfig = field(default_factory=DataConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    augment: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    seed: int = SEED
    output_dir: str = "outputs"

    def to_dict(self) -> dict:
        return asdict(self)


def add_common_args(parser):
    """Attach the CLI flags shared by every entry-point script."""
    parser.add_argument("--data-root", type=str, default=DataConfig.root,
                        help="Root of the preprocessed dataset (ImageFolder layout).")
    parser.add_argument("--mode", type=str, default="three_class",
                        choices=["three_class", "binary"])
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--input-size", type=int, default=ModelConfig.input_size)
    parser.add_argument("--num-workers", type=int, default=DataConfig.num_workers)
    parser.add_argument("--no-tta", action="store_true",
                        help="Disable the +/-3 deg test-time augmentation.")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Skip ImageNet weight download (offline / air-gapped runs). "
                             "The manuscript initialises every model from ImageNet weights, "
                             "so this should be left off when reproducing results.")
    return parser


def config_from_args(args) -> Config:
    """Build a Config from parsed CLI arguments, keeping manuscript defaults."""
    cfg = Config()
    cfg.data.root = args.data_root
    cfg.data.mode = args.mode
    cfg.data.num_workers = args.num_workers
    cfg.output_dir = args.output_dir
    cfg.seed = args.seed
    cfg.train.epochs = args.epochs
    cfg.train.batch_size = args.batch_size
    cfg.train.lr = args.lr
    cfg.model.input_size = args.input_size
    if getattr(args, "no_tta", False):
        cfg.eval.use_tta = False
    if getattr(args, "no_pretrained", False):
        cfg.model.pretrained = False
    return cfg
