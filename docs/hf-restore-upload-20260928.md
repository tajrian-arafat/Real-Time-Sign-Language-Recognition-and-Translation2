# Hugging Face cloud restore re-upload (2026-09-28)

Re-published the **~66.95% overnight** ASL checkpoint to dataset [`Taalvi/sign-language-cloud-restore`](https://huggingface.co/datasets/Taalvi/sign-language-cloud-restore) as the single source of truth for restore + deploy.

## Checkpoint

| Field | Value |
|--------|--------|
| Path used | `models/kaggle_overnight_v1/best.pt` |
| Provenance | Copied from current HF dataset `best.pt` (local overnight dir was empty; local `kaggle_extended_v1/best.pt` was **64.04%** epoch 15, different SHA) |
| Checkpoint metadata `best_val_acc` | **0.6695637138663053** |
| Epoch | 18 |
| `run_mode` | `kaggle_processed` |

## Fresh validation metrics (`python -m ml.evaluate --split val`)

Evaluated on **9,211** val samples, **250** classes, real processed tensors (`SIGN_LANGUAGE_DATA_ROOT=/workspace/data`).

| Metric | Value |
|--------|--------|
| Top-1 | **0.669526** |
| Top-5 | **0.871132** |
| Macro F1 | 0.655297 |

Full report: `reports/metrics.json` (generated 2026-09-28T12:24:58Z).

## Bundle

Built with `PYTHONPATH=/workspace python scripts/build_cloud_restore.py` after `python -m ml.export_onnx --checkpoint models/kaggle_overnight_v1/best.pt`.

Contents under `artifacts/cloud_restore/`:

- `model.onnx`
- `best.pt`
- `label_map.json`
- `splits/train_tensor_index.json`, `splits/val_tensor_index.json`, `splits/test_tensor_index.json`
- `MANIFEST.json` (`extended_checkpoint.best_val_acc`: **0.6695637138663053**)

## Hugging Face upload

- Script: `bash scripts/upload_cloud_restore_hf.sh`
- **Commit:** https://huggingface.co/datasets/Taalvi/sign-language-cloud-restore/commit/b586a64d1c60f394089dfc02980580d5e6c63d35

## Git (tooling only)

Added `scripts/build_cloud_restore.py` from branch `cursor/overnight-cpu-train-9cb6` on `cursor/hf-overnight-67-reupload-4a21`. No `.pt` / `.onnx` committed.
