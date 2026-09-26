"""Detect Kaggle API credentials without exposing secrets."""
from __future__ import annotations

import os
from pathlib import Path


def kaggle_credentials_present() -> bool:
    """True if legacy username/key, KAGGLE_API_TOKEN, or ~/.kaggle/access_token is set."""
    if os.environ.get("KAGGLE_USERNAME", "").strip() and os.environ.get(
        "KAGGLE_KEY", ""
    ).strip():
        return True
    if os.environ.get("KAGGLE_API_TOKEN", "").strip():
        return True
    token_path = Path.home() / ".kaggle" / "access_token"
    if token_path.is_file() and token_path.read_text(encoding="utf-8").strip():
        return True
    return False


def kaggle_credentials_message() -> str:
    return (
        "Set KAGGLE_USERNAME and KAGGLE_KEY, or export KAGGLE_API_TOKEN "
        "(see https://www.kaggle.com/settings — API token), then re-run."
    )


def kaggle_competition_entered(competition_slug: str = "asl-signs") -> tuple[bool | None, str]:
    """
    Return (user_has_entered, detail). None if the check could not run.
    Download requires accepting competition rules on kaggle.com (Join Competition).
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        return None, "kaggle package not installed"

    api = KaggleApi()
    api.authenticate()
    response = api.competitions_list(search=competition_slug)
    competitions = getattr(response, "competitions", None) or []
    for comp in competitions:
        ref = str(getattr(comp, "ref", "") or "")
        if competition_slug in ref:
            entered = bool(
                getattr(comp, "user_has_entered", None)
                or getattr(comp, "userHasEntered", False)
            )
            detail = "user_has_entered=True" if entered else "user_has_entered=False"
            return entered, detail
    return None, f"competition {competition_slug!r} not found in list"
