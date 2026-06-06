# PigBDMR Training Reference

This file records the default PigBDMR training command. The public repository does not include training annotations; replace `/path/to/train.jsonl` with your own training file.

`--num_virtual_tokens` controls the number of Virtual Encoder tokens. `--num_dummies` remains a backward-compatible legacy alias.

```bash
conda run -n pigbdmr --no-capture-output python PigBDMR/train.py data/MR.py \
  --exp_id pigbdmr_train \
  --use_neg --dset_name hl --ctx_mode video_tef \
  --train_path /path/to/train.jsonl \
  --eval_path data/QV-M2/test.jsonl \
  --v_feat_dirs features/slowfast_features features/clip_features \
  --v_feat_dim 2816 \
  --t_feat_dir features/clip_text_features_new \
  --t_feat_dim 512 \
  --max_v_l 75 --max_q_l 40 --max_windows 5 \
  --bsz 64 --n_epoch 150 --eval_bsz 1 --eval_epoch 3 \
  --use_SRM --use_pv_repr \
  --kernel_size 5 --num_conv_layers 1 --num_mlp_layers 5 \
  --t2v_layers 6 --num_virtual_tokens 40 \
  --lw_reg 1.0 --lw_cls 5.0 --lw_saliency 0.8 \
  --lw_pv 9.0 --lw_pv1 0.7
```

## Notes

- `features/` must contain the downloaded SlowFast, CLIP visual, and CLIP text features.
- Training outputs are written to `results/`, which is ignored by Git.
- Evaluation uses the public test annotations in `data/QV-M2/test.jsonl`.
