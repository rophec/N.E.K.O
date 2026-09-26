# Mahjong YOLO Dataset Registry

The current derived dataset and its immediate reproducibility baseline are kept here.

## Current Training Dataset

- `15_balanced_dragon_features_20260713/`
- Train: 1236 images, 55372 boxes
- Validation: 311 images, 17110 boxes
- Split ratio: 79.90% train / 20.10% validation
- Includes the previous training data plus balanced 5z/6z real-slot feature samples.
- Train synthetic samples use only canonical train backgrounds; validation synthetic samples use only canonical validation backgrounds.
- Does not contain the fixed test split.

## Previous Baseline

- `14_real_slot_synthetic_20260712/`
- Train: 836 images
- Validation: 11 images
- Retained only to reproduce the previous model; do not use for the next training run.

## Canonical Manual Ground Truth

The immutable manually verified source is stored outside this derived registry:

`../manual_ground_truth_canonical_20260712/`

- Train: 42 images
- Validation: 11 images
- Test: 54 images
- Contains matching YOLO HBB TXT and X-AnyLabeling JSON annotations.

## Training Entry

Use `tools/mahjong_yolo/train_yolo26m_batch6_combined_test.py`. Its defaults point to the balanced current dataset and canonical fixed test.
