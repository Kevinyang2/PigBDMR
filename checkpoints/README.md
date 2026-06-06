# Checkpoints

Model checkpoints are not stored in Git and are not included in the PigMMR Baidu Netdisk dataset package. If checkpoints are released separately, place them under this directory.

Recommended layout:

```text
checkpoints/
  pigbdmr/
    model_best.ckpt
    opt.json
```

`model_best.ckpt` contains model weights. `opt.json` records the training configuration and is useful for reproducing inference settings.

Checkpoint files such as `.ckpt`, `.pth`, and `.pt` are ignored by Git.
