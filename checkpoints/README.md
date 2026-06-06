# Checkpoints

Model checkpoints are not stored in Git. Download them from the project release page or model card, then place them under this directory.

Recommended layout:

```text
checkpoints/
  pigbdmr/
    model_best.ckpt
    opt.json
```

`model_best.ckpt` contains model weights. `opt.json` records the training configuration and is useful for reproducing inference settings.

Checkpoint files such as `.ckpt`, `.pth`, and `.pt` are ignored by Git.
