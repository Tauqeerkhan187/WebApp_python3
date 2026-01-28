# Author: TK
# Date: 28-01-2026
# Desc: Loads the app state from JSON, prevents app from crashing, writes
# to temp file, fsync.

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict

DEFAULT_PATH = "store.json"


def load_state(path: str = DEFAULT_PATH) -> Dict[str, Any]:
    """
    Load the app state from JSON.

    Returns {} if file doesnt exist yet, file is corrupted, or r permission
    issue happen.

    """
    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
            return data if isinstance(data, dict) else {}

    except (json.JSONDecodeError, OSError):
        return {}


def save_state(state: Dict[str, Any], path: str = DEFAULT_PATH) -> None:
    """
    writes to temp file, fsync, os.replace to swap into place, this avoids
    corrupt store,json if program crashes mid-write.
    """

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    payload = json.dumps(state, indent=2, ensure_ascii=False, sort_keys=True)

    dir_name = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".store_", suffix=".json", dir=dir_name)

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
            tmp.write(payload)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, path)

    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass



