# Real-Time ASL Recognition and Bangla Translation

Isolated-sign ASL recognition with English gloss accumulation and Bangla translation. See [docs/ASL-Bangla-System-Architecture.md](docs/ASL-Bangla-System-Architecture.md) and [docs/Cursor-Master-Prompt-and-Agents.md](docs/Cursor-Master-Prompt-and-Agents.md).

## Data directory (Windows)

On the author's Windows PC, **all dataset and large downloads must live only under:**

`D:\PROJECTS\sign language`

Configure this via `config/config.yaml` → `paths.data_root`, or override at runtime:

```powershell
set SIGN_LANGUAGE_DATA_ROOT=D:\PROJECTS\sign language
```

Application code must read the data root from config/environment — **do not hardcode** this path in Python/TypeScript source.

On this cloud development VM, use repo-relative storage under `/workspace/data/` by setting:

```bash
export SIGN_LANGUAGE_DATA_ROOT=/workspace/data
```

Kaggle `asl-signs` download accepts either legacy `KAGGLE_USERNAME` / `KAGGLE_KEY` or the newer `KAGGLE_API_TOKEN` (also read from `~/.kaggle/access_token`). Never commit tokens.

(or leave unset once path resolution helpers are added in later agents; the config file documents both patterns).

## Quick start (Agent 1 scaffolding)

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
python scripts/probe_environment.py
node --version
```

Full training, inference, and evaluation commands will be documented here as later agents land.
