# Gazelle Liver Segmentation

This repository contains a teacher-student pipeline for 3D liver segmentation and reconstruction from medical CT volumes. The project was developed for the 2026 China College Students' Integrated Circuit Innovation and Entrepreneurship Competition, commonly referred to as the China College IC Competition.

The core idea is to first train a 3D U-Net teacher model for volumetric organ segmentation, then distill its predictions into a compact Gazelle-oriented student model with low-bit quantized linear layers. The student model is used to reconstruct a 3D liver mask and export deployment-friendly parameters.

## Features

- Preprocesses NIfTI CT volumes with orientation normalization, spacing resampling, intensity normalization, and body-region cropping.
- Trains a patch-based 3D U-Net teacher segmentation model.
- Runs teacher inference with sliding-window prediction, test-time augmentation, threshold search, and post-processing.
- Converts teacher predictions into sampled ROI features for student training.
- Trains a two-layer quantized MLP student model using teacher distillation.
- Exports Gazelle-friendly linear parameters and model metadata.
- Reconstructs 3D liver masks, probability arrays, OBJ meshes, and preview images from student predictions.

## Repository Structure

```text
.
├── preprocess_teacher_dataset.py       # Preprocess raw Task03_Liver-style NIfTI data
├── train_teacher_3d_segmentation.py    # Train the 3D U-Net teacher model
├── infer_teacher_3d_segmentation.py    # Run teacher inference and save probability maps
├── prepare_student_from_teacher.py     # Build student training samples from teacher outputs
├── train_student_for_gazelle.py        # Train and export the quantized student model
├── reconstruct_with_student.py         # Reconstruct 3D liver masks from student predictions
├── reconstruct_with_student_best.py    # Reconstruct using best scanned post-processing parameters
├── teacher_common.py                   # Shared preprocessing utilities
├── teacher_dataset.py                  # Teacher datasets and patch sampling
├── teacher_models.py                   # 3D U-Net implementation
├── teacher_metrics.py                  # Dice, IoU, precision, recall, and loss utilities
└── student_recon_common.py             # Student feature and ROI helpers
```

## Expected Data Layout

By default, the scripts expect a Medical Segmentation Decathlon Task03_Liver-style dataset layout:

```text
Task03_Liver/
├── imagesTr/
│   └── *.nii.gz
└── labelsTr/
    └── *.nii.gz
```

Generated outputs are written under `outputs2/` by default. Datasets, checkpoints, model weights, and generated outputs are intentionally excluded from version control.

## Quick Start

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Run the full pipeline:

```bash
python preprocess_teacher_dataset.py
python train_teacher_3d_segmentation.py
python infer_teacher_3d_segmentation.py --split both
python prepare_student_from_teacher.py
python train_student_for_gazelle.py
python reconstruct_with_student_best.py
```

Each script exposes command-line arguments for overriding data paths, output paths, training hyperparameters, thresholds, and reconstruction settings.

## Outputs

The pipeline can produce:

- `teacher_best.pt` and `teacher_last.pt` teacher checkpoints.
- Teacher probability maps and binary masks for training and validation cases.
- Student-ready sampled features, labels, and teacher probabilities.
- Quantized student model parameters:
  - `optical_params.pkl`
  - `nonlinear_params.pkl`
  - `gazelle_optical_params.pkl`
  - `gazelle_model_meta.pkl`
- Reconstructed liver masks in NIfTI format.
- OBJ surface meshes and PNG preview images.

## Acknowledgements

We thank Zhengquan Jiang for his collaboration on this project.
