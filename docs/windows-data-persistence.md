# Persist all project data on Windows (`D:\PROJECTS\sign language`)

This is the **canonical local data root** for Tajrian’s machine (see `config/config.yaml`). Cloud agents use `/workspace/data` unless you point them at a synced copy.

## One-time setup (PowerShell, run as yourself)

From the repo root:

```powershell
$env:SIGN_LANGUAGE_DATA_ROOT = 'D:\PROJECTS\sign language'
.\scripts\windows\Ensure-DataRoot.ps1
.\scripts\windows\Run-KaggleAcquisition.ps1
```

Requires:

- Python 3.12+ and `pip install -r requirements.txt`
- Kaggle: `KAGGLE_API_TOKEN` in env **or** `%USERPROFILE%\.kaggle\access_token`
- Competition rules accepted: https://www.kaggle.com/competitions/asl-signs/rules

## What gets stored under `D:\PROJECTS\sign language`

| Path | Keep? | Purpose |
|------|--------|---------|
| `raw/kaggle_asl_signs/` | **Yes** | ~38 GB zip + ~94k parquets — **never re-download** if intact |
| `processed/kaggle_asl_signs/` | **Yes** | `.npz` tensors, splits, `label_map.json` |
| `models/` (copy from repo) | **Yes** | Copy `models/kaggle_baseline/best.pt`, `models/served/model.onnx`, `bangla_dictionary.json` here optionally via `scripts/windows/Sync-ServedModels.ps1` |

**Do not** put secrets in this folder. Kaggle token stays in env or `~/.kaggle/`.

## After a cloud training run

Copy served artifacts from the cloud VM (or snapshot pod) into the repo or data root:

```powershell
$env:SIGN_LANGUAGE_DATA_ROOT = 'D:\PROJECTS\sign language'
.\scripts\windows\Sync-ServedModels.ps1 -SourceDir 'C:\path\to\downloaded\models'
```

## Rerun policy (avoid full redo)

1. If `raw/kaggle_asl_signs` has `train.csv` and ~94,477 parquets → **skip download** (`download_kaggle_islr.py` exits 0 with SKIP).
2. If `processed/.../tensors/<hash>/` is complete → **skip preprocess** unless preprocess code changed.
3. Training only: `python -m ml.train --resume models/kaggle_baseline/last.pt ...`

## Local Cursor agent (self-hosted)

To let a cloud coordinator write directly to `D:\PROJECTS\sign language`, run **Cursor self-hosted worker** on your PC with the repo checked out, then ask the Project to “download Kaggle data to SIGN_LANGUAGE_DATA_ROOT on my machine.”

Without a connected worker, only **you** (or these scripts) can populate `D:\`.
