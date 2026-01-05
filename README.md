# ECA-FedNet: Clinically Validated Privacy-Preserving Attention-Enhanced Deep Learning for Cardiovascular Disease Diagnosis from Paper-Based ECG

This repository provides the official implementation and documentation for ECA-FedNet, a clinically validated deep learning framework designed for the diagnosis of cardiovascular diseases from paper-based electrocardiogram (ECG) images. The framework integrates robust image preprocessing, attention-enhanced convolutional modeling, explainable artificial intelligence, and personalized federated learning to enable accurate, interpretable, and privacy-preserving diagnostics in real-world clinical environments.

This repository accompanies the corresponding research article and is intended to support reproducibility, methodological transparency, and future research extensions.

## 1. Background and Motivation

Electrocardiography remains one of the most widely used diagnostic tools for cardiovascular disease. However, in many healthcare systems—particularly in low- and middle-income regions—ECG records are still stored and archived in paper-based formats. These analog ECGs are often digitized through scanning or photography, introducing significant variability due to grid artifacts, faded ink, lighting conditions, scanning noise, and handwritten annotations.

Conventional deep learning approaches, particularly those designed for clean digital ECG signals, fail to generalize effectively to such heterogeneous and noisy image-based ECG data. Additionally, centralized data aggregation for model training is often infeasible in healthcare due to privacy regulations and ethical constraints.

ECA-FedNet addresses these challenges by introducing:

  • An attention-enhanced deep learning architecture tailored for noisy paper-based ECG images.

  • A privacy-preserving federated learning framework that enables collaborative model training without sharing raw patient data.

  • Clinically validated explainability mechanisms to support medical trust and deployment.

## 2. Contributions

The key contributions of this work are summarized as follows:

  • Development of a curated and clinically validated dataset of paper-based ECG images collected from a tertiary-care hospital.

  • Design of a robust nine-step preprocessing pipeline to suppress grid artifacts and standardize heterogeneous ECG images.

  • Proposal of ECA-Net, a dual-attention convolutional neural network based on EfficientNet-B3 for fine-grained ECG morphology learning.

  • Extension to ECA-FedNet, a personalized federated learning framework compliant with data privacy regulations.

  • Integration of Grad-CAM–based explainability validated by clinical experts.

  • Deployment of an end-to-end diagnostic system through the CardioCare AI agent.

## 3. Dataset Description

  • Source: Shaheed Suhrawardy Medical College and Hospital (SSMCH), Dhaka, Bangladesh

  • Total Samples: 849 paper-based 12-lead ECG images

  • Clinical Classes:
  
    • Normal: 434 images
    • Abnormal: 344 images
    • Myocardial Infarction (MI): 71 images

Clinical Validation: All ECG labels were verified by a licensed medical professional.

Privacy Protection: All ECG images were fully de-identified prior to analysis.

Data Availability

    The dataset used in this study is available on request, subject to institutional approval and ethical compliance.

## 4. Preprocessing Pipeline

A dedicated preprocessing framework was designed to handle real-world degradation in paper-based ECGs. The pipeline includes:

  • Image cropping and normalization

  • Grayscale conversion

  • Threshold-based ink extraction

  • Background grid suppression

  • Signal inversion and contrast enhancement

  • Morphological dilation for waveform continuity

  • Connected component analysis for noise removal

  • Spatial centering on a uniform white canvas

  • Standardized resolution formatting

This process produces high-contrast ECG waveforms on a clean background, ensuring reliable feature extraction by convolutional neural networks.

## 5. Model Architecture
### 5.1 ECA-Net (Centralized Learning)

  • Backbone: EfficientNet-B3 (ImageNet pretrained)

  • Attention Mechanism: Convolutional Block Attention Module (CBAM)

  • Channel Attention to emphasize diagnostically relevant features

  • Spatial Attention to suppress background noise and grid artifacts

  • Optimizer: AdamW

  • Loss Function: Cross-Entropy with label smoothing

### 5.2 ECA-FedNet (Federated Learning)

Training Paradigm: Personalized Federated Learning

Data Handling: Raw ECG images remain within local institutions

Aggregation Method: Federated Averaging (FedAvg)

Objective: Enable multi-center collaboration without violating data privacy regulations such as HIPAA and GDPR

## 6. Explainable AI and Clinical Trust

To address the black-box nature of deep learning models, Grad-CAM visualizations are integrated into the framework. These attention maps highlight the ECG regions influencing model predictions and were reviewed by clinical experts to confirm physiological relevance (e.g., ST-segment morphology).

This explainability component is critical for:

Clinical validation

Trustworthy deployment

Regulatory acceptance

## 7. CardioCare AI Agent

CardioCare is a deployable clinical decision support system built on top of ECA-Net. The system:

Accepts scanned or photographed ECG images

Automatically performs preprocessing and classification

Outputs diagnostic predictions with confidence scores

Provides Grad-CAM overlays for interpretability

Achieves low-latency inference on consumer-grade hardware

## 8. Experimental Summary

Centralized ECA-Net Accuracy: 99.32%

Federated ECA-FedNet Accuracy: 97.94%

External Dataset Accuracy: 90.41%

Myocardial Infarction Sensitivity: Near-perfect in both centralized and federated settings

These results demonstrate strong robustness, generalization, and minimal performance degradation under privacy-preserving training.


## 9. License

This project is licensed under the Creative Commons Attribution–NonCommercial 4.0 International (CC BY-NC 4.0) License.

Commercial use is strictly prohibited without explicit permission from the authors.


## 10. Contact

For correspondence, data access requests, or academic collaboration:

    Sabit Ahamed Preanto
    Department of Computer Science & Engineering
    Email: preanto15-5059@diu.edu.bd
