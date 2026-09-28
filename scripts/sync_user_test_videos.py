#!/usr/bin/env python3
"""Download user ASL test videos listed in a Drive manifest (missing files only).

Agents should re-list the Drive folder and refresh the manifest before each run.
See internal/drive-asl-test-video-sync.md in the project Agent store.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    REPO
    / "artifacts"
    / "user_test_videos"
    / "drive-folder.json"
)
DEFAULT_VIDEO_DIR = REPO / "artifacts" / "user_test_videos"


def _infer_expected_sign(title: str) -> str | None:
    base = Path(title).stem
    tokens = re.split(r"[\s_\-]+", base.upper())
    for token in tokens:
        if token in {"ASL", "MP4", "720", "1080", "1440", "2160"}:
            continue
        if len(token) >= 2:
            return token.lower()
    return None


def _download_with_gdown(file_id: str, dest: Path) -> bool:
    gdown = os.environ.get("GDOWN_BIN", "gdown")
    dest.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [gdown, file_id, "-O", str(dest), "--fuzzy"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stderr or proc.stdout, file=sys.stderr)
    return proc.returncode == 0 and dest.is_file()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(os.environ.get("SIGN_LANGUAGE_TEST_VIDEO_MANIFEST", DEFAULT_MANIFEST)),
        help="JSON with folder_id and videos[] (id, title, optional expected_sign)",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path(os.environ.get("SIGN_LANGUAGE_TEST_VIDEO_DIR", DEFAULT_VIDEO_DIR)),
    )
    parser.add_argument(
        "--meta-dir",
        type=Path,
        default=None,
        help="Defaults to manifest parent (Agent store internal/user-test-videos)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned downloads without fetching",
    )
    args = parser.parse_args()
    meta_dir = args.meta_dir or args.manifest.parent

    if not args.manifest.is_file():
        print(f"sync_user_test_videos: manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    videos = data.get("videos") or []
    if not videos:
        print("sync_user_test_videos: no videos in manifest", file=sys.stderr)
        return 1

    args.video_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    failed = 0

    for entry in videos:
        file_id = str(entry.get("id", "")).strip()
        title = str(entry.get("title", "")).strip()
        if not file_id or not title:
            print(f"sync_user_test_videos: skip invalid entry {entry!r}", file=sys.stderr)
            failed += 1
            continue

        dest = args.video_dir / title
        if dest.is_file():
            skipped += 1
            continue

        if args.dry_run:
            print(f"would download {title} ({file_id})")
            continue

        ok = _download_with_gdown(file_id, dest)
        if not ok:
            print(
                f"sync_user_test_videos: gdown failed for {title}; "
                "use Google Drive MCP download_file_content for this id.",
                file=sys.stderr,
            )
            failed += 1
            continue
        downloaded += 1

        expected = entry.get("expected_sign") or _infer_expected_sign(title)
        meta = {**entry, "expected_sign": expected, "local_path": str(dest)}
        safe = title.replace("/", "_")
        (meta_dir / f"{safe}.meta.json").write_text(
            json.dumps(meta, indent=2) + "\n",
            encoding="utf-8",
        )

    args.manifest.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(
        f"sync_user_test_videos: downloaded={downloaded} skipped={skipped} failed={failed}",
        file=sys.stderr,
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
