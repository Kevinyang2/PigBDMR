# Data

The PigMMR test annotations can be downloaded from Baidu Netdisk:

```text
Link: https://pan.baidu.com/s/1nb4DeaZHwt0ie0kQ5_F0FQ
Extraction code: best
```

The downloaded `data` directory contains:

```text
data/
  test.jsonl
  video_chunks.csv
```

This repository includes a copy of the public test annotations used for evaluation:

```text
data/QV-M2/test.jsonl
```

Training annotations are not included in the Git repository. To train PigBDMR, prepare a training JSONL file with the same schema and pass it to `PigBDMR/train.py` with `--train_path`.

## Annotation Schema

Each line is a JSON object:

```json
{
  "qid": 114,
  "query": "A pig eats from the feeder.",
  "duration": 149.769,
  "vid": "multi_0007",
  "relevant_clip_ids": [0, 1, 2],
  "saliency_scores": [[4, 4, 4]],
  "relevant_windows": [[0.0, 15.089], [25.925, 74.082]]
}
```

Required fields for moment retrieval evaluation are:

- `qid`: query id.
- `query`: natural-language behavior query.
- `vid`: video id used to locate pre-extracted features.
- `duration`: video duration in seconds.
- `relevant_windows`: list of ground-truth temporal windows in seconds.

`relevant_clip_ids` and `saliency_scores` are used by highlight-style evaluation utilities when available.

## Feature Mapping

Feature files are expected to be named by `vid` and stored under the feature directories described in `features/README.md`.
