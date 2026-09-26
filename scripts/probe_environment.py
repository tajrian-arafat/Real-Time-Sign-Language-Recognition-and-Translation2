#!/usr/bin/env python3
"""Write reports/environment.json from current machine (Agent 1)."""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def disk_free_gb(path: str) -> float:
    usage = shutil.disk_usage(path)
    return round(usage.free / (1024**3), 2)


def gpu_info() -> dict:
    info: dict = {"present": False, "devices": []}
    try:
        import torch

        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["present"] = True
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info["devices"].append(
                    {
                        "index": i,
                        "name": props.name,
                        "vram_gb": round(props.total_memory / (1024**3), 2),
                    }
                )
        else:
            info["note"] = "torch.cuda.is_available() is False"
    except ImportError:
        info["note"] = "torch not installed; GPU probe skipped"
    return info


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    node_version = None
    if shutil.which("node"):
        node_version = subprocess.check_output(["node", "--version"], text=True).strip()

    gpu = gpu_info()
    payload = {
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "cpu_cores_logical": os.cpu_count(),
        "ram_total_gb": round(
            int(Path("/proc/meminfo").read_text().split("MemTotal:")[1].split()[0])
            / (1024**2),
            2,
        )
        if Path("/proc/meminfo").exists()
        else None,
        "disk_free_gb_workspace": disk_free_gb(str(root)),
        "gpu": gpu,
        "node_version": node_version,
        "recommended_training": {
            "device": "cuda" if gpu.get("present") else "cpu",
            "batch_size": 128,
            "num_workers": max(1, (os.cpu_count() or 2) - 1),
        },
        "verification": {
            "python_version_cli": subprocess.check_output(
                [sys.executable, "--version"], text=True
            ).strip(),
            "node_version_cli": node_version,
        },
    }

    out = reports / "environment.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
