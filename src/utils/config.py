"""YAML configuration loading with shell-style environment variable expansion.

Supported syntax inside any string value of the YAML file:

* ``${VAR}``            -> value of ``VAR`` (error if unset)
* ``${VAR:-default}``   -> value of ``VAR`` if set and non-empty, else ``default``

Defaults may themselves contain variables, e.g.
``${HF_HOME:-${SCRATCH:-./scratch}/.cache/huggingface}``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "configs" / "config.yaml"

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^${}]*))?\}")


def _substitute(match: re.Match[str]) -> str:
    name, default = match.group(1), match.group(2)
    value = os.environ.get(name)
    if value:
        return value
    if default is not None:
        return default
    raise KeyError(f"Environment variable '{name}' is not set and has no default in config.")


def expand_env(value: str) -> str:
    """Expand ``${VAR}`` / ``${VAR:-default}`` patterns, innermost first."""
    previous = None
    while previous != value:
        previous = value
        value = _ENV_PATTERN.sub(_substitute, value)
    return value


def _expand_recursive(node: Any) -> Any:
    if isinstance(node, dict):
        return {key: _expand_recursive(val) for key, val in node.items()}
    if isinstance(node, list):
        return [_expand_recursive(val) for val in node]
    if isinstance(node, str):
        return expand_env(node)
    return node


def load_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load a YAML config and expand environment variables in all string values."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return _expand_recursive(raw)


def resolve_path(path: str | os.PathLike[str], base: str | os.PathLike[str] | None = None) -> Path:
    """Return ``path`` as an absolute path.

    Relative paths are interpreted relative to ``base`` (default: project root), so
    scripts behave identically regardless of the working directory they are launched from.
    """
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (Path(base) if base is not None else PROJECT_ROOT).joinpath(candidate).resolve()
