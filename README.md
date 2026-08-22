# ECA-Net / ECA-FedNet

**A Privacy-Aware Attention-Based Deep Learning Framework for Cardiovascular Disease Diagnosis from Paper-Based ECG**

Reference implementation for the manuscript of the same name. The repository reproduces every experiment reported in the paper: the preprocessing pipeline, the ECA-Net classifier, the ECA-FedNet federated extension, the transfer-learning comparison, the attention ablation, the complexity analysis, Grad-CAM interpretability, cross-domain evaluation, and the CardioCare inference prototype.

ECA-Net pairs an EfficientNet-B3 backbone with a sequential channel-then-spatial attention module (CBAM) to emphasise diagnostic waveform features while suppressing grid lines, ink fading and scanning artifacts in digitised paper ECGs. ECA-FedNet extends it to a multi-institution setting in which only model parameters — never ECG records — cross institutional boundaries.

---

## Headline results

| Model | Protocol | Accuracy | Balanced accuracy | Macro-F1 | κ | MCC |
|---|---|---|---|---|---|---|
| ECA-Net | A — pooled out-of-fold (n = 1000) | 86.40% | 86.28% | 0.8621 | 0.7890 | 0.7890 |
| ECA-FedNet | B — held-out test (n = 200) | 83.00% | 82.88% | 0.8262 | 0.7368 | 0.7371 |

The attention module contributes 295,010 parameters (2.68% of the network) and no measurable latency: both ECA-Net and the plain EfficientNet-B3 run at 4.5 ms per image at 300 × 300 on a Tesla T4.

**The two protocols are not directly comparable** — they use different data splits and different amounts of training data. The paper is explicit on this point, and so is this repository. Use `--centralized-baseline` in the federated script for a matched comparison on an identical split.

---

## Repository layout

```
eca-net/
├── ecanet/                      # library
│   ├── config.py                # every hyperparameter, keyed to a manuscript table
│   ├── preprocessing.py         # Table I image pipeline (11 steps)
│   ├── data.py                  # datasets, augmentation, sampling, splits, client partitioning
│   ├── models.py                # ECA-Net, CBAM, ablation variants, baseline builders
│   ├── losses.py                # class-balanced focal + inverse-sqrt weighted CE
│   ├── engine.py                # training loop, TTA inference, complexity profiling
│   ├── federated.py             # FedAvg aggregation and client updates
│   ├── metrics.py               # all reported metrics with confidence intervals
│   ├── gradcam.py               # interpretability maps
│   └── visualization.py         # figure generation
├── scripts/                     # entry points
│   ├── train_ecanet.py          # Protocol A, 5-fold OOF        → Table XI, Figs. 7(g), 8
│   ├── train_ecafednet.py       # Protocol B, FedAvg            → Tables V, XI, Figs. 7(h), 9
│   ├── run_benchmark.py         # transfer-learning comparison  → Table X
│   ├── run_ablation.py          # attention ablation            → Table XII
│   ├── run_complexity.py        # size / FLOPs / latency        → Tables IV, VI, VII, VIII
│   ├── run_gradcam.py           # interpretability              → Fig. 10
│   ├── run_external_validation.py  # cross-domain transfer      → Table XIII
│   └── predict.py               # CardioCare inference CLI      → Figs. 11-12
├── requirements.txt
└── README.md
```

---

## Installation

Python 3.10+ and a CUDA-capable GPU are recommended (experiments were run on a Tesla T4).

```bash
git clone https://github.com/<user>/eca-net.git
cd eca-net
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`albumentations` is pinned below 2.0: the 2.x series renames the `ShiftScaleRotate` and `GaussNoise` arguments used by the Table II augmentation policy.

`thop` is optional — GFLOPs are simply reported as empty if it is unavailable.

---

## Dataset

**SSMCH-ECG** — 1,000 paper-based ECG records collected at Shaheed Suhrawardy Medical College and Hospital, Dhaka, Bangladesh, under ethical approval REC-FSIT-2025/No: 11474: 434 Normal, 344 Abnormal, 222 Myocardial Infarction. All records were reviewed and labelled by a clinical expert, and manually de-identified before digitisation — headers carrying names, hospital identifiers, acquisition dates and physician details were removed, and no linkage to hospital records was retained.

Per the manuscript's Data Availability Statement, the dataset is **available from the corresponding author on reasonable request**. It is not distributed with this repository, and `.gitignore` excludes image files so records cannot be committed accidentally.

Expected layout (torchvision `ImageFolder`):

```
data/SSMCH-ECG/
├── Abnormal/
├── MI/
└── Normal/
```

Folder names sort alphabetically to Abnormal = 0, MI = 1, Normal = 2 — the class order used throughout the manuscript's tables.

### Preprocessing

Raw scans are converted to model-ready images with the eleven-step pipeline of Table I:

```bash
python -m ecanet.preprocessing \
    --input  data/raw \
    --output data/SSMCH-ECG \
    --header-frac 0.12          # optional automatic header removal
```

| # | Step | Value |
|---|---|---|
| 1 | Manual ROI cropping | removes header / patient identifiers |
| 2 | Grayscale conversion | ITU-R BT.601 luma |
| 3 | Intensity thresholding | T = 60 |
| 4 | Ink-mask extraction | binary |
| 5 | Canvas normalisation | 3000 × 3000 px, white, aspect ratio preserved |
| 6 | Grayscale inversion | — |
| 7 | Morphological dilation | 3 × 3 kernel, 1 iteration |
| 8 | Connected-component filtering | minimum area 50 px |
| 9 | Lossless export | PNG |
| 10 | Ink-bounding-box crop | grey threshold < 245, 3% padding |
| 11 | Resize | 300 × 300 |

Step 1 is semi-automatic in the paper. `--header-frac` provides a fractional fallback for batch processing; pass `0.0` when headers have already been removed. The threshold in step 3 was chosen by sweeping T over {40, 50, 60, 70, 80} — balanced accuracy varied by under 1.5 percentage points, so the pipeline is not critically sensitive to it.

---

## Reproducing the paper

### ECA-Net — Protocol A (Table XI, Figs. 7g, 8)

Stratified five-fold cross-validation over the 1,000 original records. Augmentation is applied only to the training portion of each fold and only on the fly; validation folds contain unmodified originals. The five sets of out-of-fold predictions are pooled, so every record receives exactly one prediction from a model that never saw it or any augmented version of it.

```bash
python scripts/train_ecanet.py --data-root data/SSMCH-ECG --mode three_class --save-checkpoints
```

### ECA-FedNet — Protocol B (Tables V, XI, Fig. 9)

A fixed stratified 80/20 split removes 200 test records *before* any client is formed. The remaining 800 are partitioned across four simulated institutions by a per-class Dir(α = 0.4) draw, producing both label skew and quantity skew. Twenty-five FedAvg rounds, two local epochs each, full participation.

```bash
python scripts/train_ecafednet.py --data-root data/SSMCH-ECG
python scripts/train_ecafednet.py --data-root data/SSMCH-ECG --centralized-baseline
```

The second command additionally trains a centralized model on the identical split with a matched compute budget (T × E epochs), which supplies the matched comparison the manuscript lists as missing in Section V-E.

### Transfer-learning comparison (Table X)

```bash
python scripts/run_benchmark.py --data-root data/SSMCH-ECG
```

All architectures share the same split, augmentation, optimiser, schedule and test-time augmentation, and are ranked by balanced accuracy. Add `--include-extra` for the screened-but-unreported models (EfficientNet-B0, DenseNet-121, ConvNeXt-Tiny).

### Attention ablation (Table XII)

```bash
python scripts/run_ablation.py --data-root data/SSMCH-ECG
```

Trains M_base (no attention), M_CA (channel only), M_SA (spatial only) and M_dual (proposed) under identical conditions, and reports the sub-additivity of the two branches discussed in Section IV-G.

### Complexity (Tables IV, VI, VII, VIII)

```bash
python scripts/run_complexity.py
```

Requires no data. Latency follows Section IV-B: batch size 8, four warm-up iterations, 20 timed forward passes.

### Grad-CAM (Fig. 10)

```bash
python scripts/run_gradcam.py --data-root data/SSMCH-ECG \
    --checkpoint outputs/ecanet_three_class_dual/ecanet_fold1.pth
```

Maps are drawn from the final convolutional stage of the backbone; `--after-attention` hooks the CBAM output instead. **These visualisations are qualitative.** They were not scored against clinician-annotated diagnostic regions and no localisation metric was computed, so they illustrate model behaviour rather than demonstrate localisation accuracy (Section V-E).

### Cross-domain evaluation (Table XIII)

```bash
python scripts/run_external_validation.py \
    --external-root data/external-ecg \
    --checkpoint outputs/ecanet_three_class_dual/ecanet_fold1.pth
```

The external corpus (ECG Images dataset of Cardiac Patients, Mendeley Data) contains a *History of MI* class absent from SSMCH-ECG, so the three-class head is replaced with a four-class head and adapted on part of the corpus, with results reported on the remainder. This measures feature transfer, **not** zero-shot generalisation, and the class-definition mismatch must be kept in view when reading the per-class figures.

### CardioCare inference (Figs. 11-12)

```bash
python scripts/predict.py --checkpoint model.pth --image scan.png --raw --gradcam
```

Returns the predicted class, confidence and an optional Grad-CAM overlay. `--raw` runs the full preprocessing pipeline first. Images are not retained.

> **CardioCare has not been clinically evaluated.** Real-world latency, out-of-distribution robustness and clinical usability remain unassessed, and none of the diagnostic results in the manuscript were produced through it. It is a research prototype, not a medical device.

---

## Configuration

Every default in `ecanet/config.py` carries a comment naming the manuscript table that fixes it. Common overrides:

| Flag | Default | Source |
|---|---|---|
| `--mode` | `three_class` | `binary` collapses Abnormal + MI |
| `--input-size` | 300 | Tables I, IV |
| `--batch-size` | 8 (×2 accumulation = 16 effective) | Table III |
| `--lr` | 3e-4 | Table III |
| `--epochs` | 40 | early stopping on balanced accuracy, patience 10 |
| `--no-tta` | off | disables the ±3° averaged-softmax TTA |
| `--no-pretrained` | off | offline runs; the paper initialises from ImageNet |
| `--clients` / `--rounds` / `--local-epochs` / `--alpha` | 4 / 25 / 2 / 0.4 | Table IX |

### Training parameters (Table III)

| Parameter | Value |
|---|---|
| Optimizer | AdamW, weight decay 1e-4 |
| Learning rate | 3e-4, ReduceLROnPlateau (factor 0.2, patience 3) on validation balanced accuracy |
| Batch size | 16 effective |
| Loss | class-balanced focal, β = 0.99, γ = 2.0, label smoothing 0.05 |
| Sampling | weighted random sampler, weight ∝ 1/√(class frequency) |
| Precision | AMP for training, FP32 for inference |

### Augmentation (Table II) — training folds only, applied on the fly

| Technique | Parameters |
|---|---|
| Shift / Scale / Rotate | 0.03 / 0.07 / 4°, p = 0.6, reflect border |
| Gaussian noise | var (5.0, 30.0), p = 0.3 |
| Gaussian blur | limit (3, 5), p = 0.2 |
| CLAHE | p = 0.2 |
| Brightness / Contrast | 0.12 / 0.12, p = 0.5 |

Class weights and sampler weights are recomputed independently for each training fold and each federated client, so no validation data and no other client's distribution influences optimisation.

### Federated configuration (Table IX)

| Parameter | Value |
|---|---|
| Clients (K) | 4 |
| Communication rounds (T) | 25 |
| Local epochs (E) | 2 |
| Participation | 100%, every round |
| Partition | Dirichlet label skew, α = 0.4 |
| Aggregation | FedAvg weighted by N_k; integer buffers taken from the first client |
| Local optimizer | AdamW, lr 1e-3, weight decay 1e-4 |
| Local loss | CE, weights ∝ 1/√n_c (local), label smoothing 0.1 |

Each client transmits roughly 42 MB (11.0 M parameters) per round. No gradients, activations or raw images are exchanged.

---

## Verification

The implementation reproduces the manuscript's structural figures exactly:

**Parameter distribution (Table VII)** — backbone 10,696,232 · channel attention 294,912 · spatial attention 98 · head 4,611 · **total 10,995,853**, with the (B, 1536, 10, 10) feature map of Table IV at 300 × 300 input.

**Complexity (Table VI)** — parameters and GFLOPs match for all seven architectures: MobileNet-V3-Large 4.21 M / 0.61 · EfficientNet-B3 10.70 M / 3.86 · ECA-Net 11.00 M / 3.86 · EfficientNet-V2-S 20.18 M / 10.81 · ResNet-34 21.29 M / 9.61 · Swin-Tiny 27.52 M / 5.95 · ViT-B/16 85.80 M / 22.57. Latency depends on hardware; the paper's 4.5 ms figures are Tesla T4 measurements.

Seeds are fixed at 42 throughout (`seed_everything` covers Python, NumPy and PyTorch, and sets cuDNN deterministic). Exact reproduction across different GPUs, driver versions or PyTorch releases is not guaranteed — non-deterministic CUDA kernels and AMP will introduce small run-to-run variation.

---

## Known discrepancies in the manuscript

Three places where the paper is internally inconsistent. Each is handled with an explicit switch rather than a silent choice; the defaults follow the source that matches the released training code.

1. **Centralized loss.** Table III describes cross-entropy with label smoothing 0.1. Section III-E and Algorithm 1 (line 13) describe a class-balanced focal loss with β = 0.99, γ = 2.0 and label smoothing 0.05. The default is the focal loss; `--loss weighted_ce` selects the Table III variant.

2. **Federated local learning rate.** Table IX gives 1e-3; Algorithm 3 (line 5) gives 1e-4. The default is 1e-3, overridable with `--local-lr`.

3. **Memory footprint.** Table VIII reports 2.7 MB for the (B, 3, 300, 300) input tensor; 3 × 300 × 300 × 4 bytes is 1.08 MB. The 614 KB figure for the (B, 1536, 10, 10) feature map is exact. `run_complexity.py` reports computed values.

**Confidence intervals.** The manuscript states that intervals are estimated from the observed proportions and corresponding sample sizes, without naming the estimator, and the published values sit between the Wald and Wilson results (e.g. Abnormal recall 85.47% [81.64–89.12] at n = 344). This repository defaults to the Wilson score interval, which has better small-sample coverage — relevant for the 222-record MI class — with `normal` (Wald) and `bootstrap` available via `EvalConfig.ci_method`. Balanced-accuracy intervals use delta-method propagation of the per-class recall variances. Point estimates are unaffected by this choice; interval endpoints may differ slightly from the published table.

---

## Limitations

Carried over from Section V-E of the manuscript, and worth reading before building on this code:

- All 1,000 records come from a single hospital, so variation in devices, print formats and scanning conditions is under-represented.
- The four federated clients are simulated from that same institution. The Dirichlet partition creates label and quantity skew but **not** the feature-level differences (scanners, print templates, paper types) that separate real hospitals.
- Federated training is not by itself a complete privacy solution. Parameter and gradient inversion attacks remain possible; differential privacy and secure aggregation were not used. The primary privacy measure here is data residency plus de-identification.
- The small number of clients and full participation limit what can be concluded about communication efficiency and the practical cost of federation.
- Removing the ECG grid also removes time and amplitude references; the effect of losing that information has not been quantified.
- Grad-CAM analysis is qualitative only.
- Generalisation is assessed on a single external corpus with partially mismatched class definitions.

---

## Citation

```bibtex
@article{preanto2025ecanet,
  title   = {A Privacy-Aware Attention-Based Deep Learning Framework for
             Cardiovascular Disease Diagnosis from Paper-Based ECG},
  author  = {Preanto, Sabit Ahamed and Bijoy, Md. Hasan Imam and Khan, Faria and
             Hasan, Md. Zahid and Tamanna, Jannatul and Uddin, Jia and Ghose, Debasish},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year    = {2025}
}
```

Update the entry with the final volume, issue, pages and DOI once available.

---

## Ethics

The dataset was collected under ethical approval REC-FSIT-2025/No: 11474 from the Research Ethics Committee, Faculty of Science and Information Technology, Daffodil International University, in collaboration with Shaheed Suhrawardy Medical College and Hospital, Dhaka. All records were de-identified before analysis.

This software is released for research purposes. It is not approved for clinical use and must not be used to inform patient care.

## License

Add a `LICENSE` file before publishing — MIT or Apache-2.0 are common choices for research code. Note that any licence you choose applies to this source code only, not to the SSMCH-ECG dataset, which remains subject to the ethical approval and access conditions described above.

## Contact

Sabit Ahamed Preanto — preanto15-5059@diu.edu.bd
Health Informatics Research Lab, Department of Computer Science and Engineering,
Daffodil International University, Dhaka, Bangladesh.
