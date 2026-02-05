from __future__ import annotations

import json
import zipfile
from dataclasses import asdict, dataclass
from io import BytesIO
from typing import Any

import joblib


@dataclass(frozen=True)
class ArtifactMeta:
    app: str
    task: str
    model_name: str
    used_columns: list[str]
    params: dict[str, Any]


def build_artifact_bytes(*, pipeline: Any, meta: ArtifactMeta) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        model_bytes = BytesIO()
        joblib.dump(pipeline, model_bytes)
        zf.writestr("model.joblib", model_bytes.getvalue())
        zf.writestr("meta.json", json.dumps(asdict(meta), indent=2))
        zf.writestr("README.txt", "Load with joblib.load('model.joblib'). Meta in meta.json.\n")
    return buffer.getvalue()

