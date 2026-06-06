# Features

Pre-extracted features are not stored in Git. Download them from Baidu Netdisk, then place them under this directory.

```text
Link: https://pan.baidu.com/s/1nb4DeaZHwt0ie0kQ5_F0FQ
Extraction code: best
```

Expected layout for the default PigBDMR setup:

```text
features/
  pig_slowfast_features/
    <vid>.npz
  pig_clip_features/
    <vid>.npz
  pig_text_features/
    <qid-or-query>.npz
```

The default training/evaluation command expects:

- `--v_feat_dirs features/pig_slowfast_features features/pig_clip_features`
- `--v_feat_dim 2816`
- `--t_feat_dir features/pig_text_features`
- `--t_feat_dim 512`

Feature files are ignored by Git because they are large generated artifacts.
