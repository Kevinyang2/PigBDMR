# Features

Pre-extracted features are not stored in Git. Download them from the project release page or model card, then place them under this directory.

Expected layout for the default PigBDMR setup:

```text
features/
  slowfast_features/
    <vid>.npz
  clip_features/
    <vid>.npz
  clip_text_features_new/
    <qid-or-query>.npz
```

The default training/evaluation command expects:

- `--v_feat_dirs features/slowfast_features features/clip_features`
- `--v_feat_dim 2816`
- `--t_feat_dir features/clip_text_features_new`
- `--t_feat_dim 512`

Feature files are ignored by Git because they are large generated artifacts.
