# Canonical Manual Ground Truth

This directory contains only manually verified Mahjong labels. It contains no augmented, synthetic, or model-predicted samples.

- `images/<split>/`: image and matching X-AnyLabeling JSON. Open this directory directly in X-AnyLabeling.
- `labels/<split>/`: matching YOLO HBB TXT labels.
- `data.yaml`: Ultralytics train/val/test definition.
- `MANIFEST.json`: counts, provenance, class counts, and image SHA-256 values.
