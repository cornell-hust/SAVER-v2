from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from saver_agent.config import SaverAgentConfig


PREPARED_SFT_METADATA_SCHEMA_VERSION = 1


def prepared_sft_metadata_path(prepared_data_path: str | Path) -> Path:
    return Path(str(prepared_data_path) + ".meta.json")


def build_prepared_sft_metadata(*, config: SaverAgentConfig) -> Dict[str, Any]:
    snapshot = config.to_dict()
    return {
        "schema_version": int(PREPARED_SFT_METADATA_SCHEMA_VERSION),
        "preview": snapshot.get("preview", {}),
        "prompt": snapshot.get("prompt", {}),
    }


def load_prepared_sft_metadata(prepared_data_path: str | Path) -> Dict[str, Any]:
    metadata_path = prepared_sft_metadata_path(prepared_data_path)
    if not metadata_path.exists():
        return {}
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_prepared_sft_metadata(
    prepared_data_path: str | Path,
    *,
    config: SaverAgentConfig,
    extra_fields: Dict[str, Any] | None = None,
) -> Path:
    metadata = build_prepared_sft_metadata(config=config)
    if extra_fields:
        metadata.update(dict(extra_fields))
    metadata_path = prepared_sft_metadata_path(prepared_data_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata_path


def ensure_prepared_sft_metadata(
    prepared_data_path: str | Path,
    *,
    config: SaverAgentConfig | None = None,
    require_config_match: bool = False,
) -> Dict[str, Any]:
    metadata_path = prepared_sft_metadata_path(prepared_data_path)
    metadata = load_prepared_sft_metadata(prepared_data_path)
    if not metadata:
        raise ValueError(
            f"Prepared SFT metadata is missing or unreadable: {metadata_path}. "
            "Regenerate the prepared JSONL before continuing."
        )

    schema_version = int(metadata.get("schema_version", 0) or 0)
    if schema_version != int(PREPARED_SFT_METADATA_SCHEMA_VERSION):
        raise ValueError(
            f"Prepared SFT metadata schema mismatch for {prepared_data_path}: "
            f"found {schema_version}, expected {PREPARED_SFT_METADATA_SCHEMA_VERSION}. "
            "Regenerate the prepared JSONL before continuing."
        )

    if require_config_match and config is not None:
        expected = build_prepared_sft_metadata(config=config)
        if metadata.get("preview") != expected.get("preview") or metadata.get("prompt") != expected.get("prompt"):
            raise ValueError(
                f"Prepared SFT metadata does not match the current preview/prompt config for {prepared_data_path}. "
                "Regenerate the prepared JSONL with the same preview/prompt settings used for training and evaluation."
            )
    return metadata
