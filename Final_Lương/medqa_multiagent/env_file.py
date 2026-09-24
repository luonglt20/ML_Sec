"""Minimal `.env` file loader -- no third-party dependency required.

Consistent with this project's zero-runtime-dependency convention (e.g.
`llm_client.py` speaks HTTP via `urllib`, not `requests`), this is a small,
self-contained parser rather than pulling in `python-dotenv` for one
convenience feature: loading a local `.env` file's `KEY=VALUE` lines into
`os.environ`, so a provider API key (`OPENAI_API_KEY`/`DEEPSEEK_API_KEY`)
doesn't have to be `export`-ed by hand every shell session before running
the CLI or the demo UI.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union


def load_env_file(path: Union[str, Path] = ".env", override: bool = False) -> bool:
    """Load `KEY=VALUE` lines from `path` into `os.environ`.

    Safe to call unconditionally at every entrypoint's startup: if `path`
    doesn't exist, this does nothing and returns `False`. Blank lines and
    lines starting with `#` are skipped; an optional leading `export ` is
    stripped (so plain shell-style `.env` files work too); values may
    optionally be wrapped in matching single or double quotes.

    By default (`override=False`), an already-exported environment
    variable always wins over a `.env` entry of the same name -- a `.env`
    file only fills in keys that aren't already set in the shell.

    Returns:
        `True` if `path` existed and was read, `False` otherwise.
    """
    env_path = Path(path)
    if not env_path.is_file():
        return False

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]

        if override or key not in os.environ:
            os.environ[key] = value

    return True
