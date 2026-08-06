# SPDX-License-Identifier: Apache-2.0
"""Bridge OptiQ's split VLM runtime into oMLX's batched VLM engine.

OptiQ models such as Mage-VL keep the language model in mlx-lm format and the
vision tower in an ``optiq_vision`` sidecar.  They therefore cannot be loaded
by ``mlx_vlm.utils.load`` even though they are fully multimodal.  This module
adapts OptiQ's language model + VisionFrontend pair to the small interface
expected by :class:`omlx.models.vlm.VLMModelAdapter`.

The dependency is optional and imported only for a checkpoint that needs this
bridge.  Native mlx-vlm OptiQ checkpoints continue through the existing path.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlx.nn as nn

# Architectures that OptiQ implements as an mlx-lm language backend plus a
# separate vision frontend, rather than as an mlx-vlm model package.
OPTIQ_VLM_BRIDGE_MODEL_TYPES = frozenset({"mage_vl"})


def _namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def needs_optiq_vlm_bridge(model_dir: str | Path) -> bool:
    """Return whether *model_dir* declares a split OptiQ VLM architecture."""
    try:
        config = json.loads((Path(model_dir) / "config.json").read_text())
    except Exception:
        return False
    return str(
        config.get("model_type", "")
    ).lower() in OPTIQ_VLM_BRIDGE_MODEL_TYPES and isinstance(
        config.get("optiq_vision"), dict
    )


class OptiqLanguageModelAdapter(nn.Module):
    """Translate mlx-vlm's ``inputs_embeds`` spelling to mlx-lm's API."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model_wrapper = model
        self.args = model.args
        self.model_type = getattr(model, "model_type", "")

    @property
    def model(self):
        return self.model_wrapper.model

    @property
    def lm_head(self):
        return self.model_wrapper.lm_head

    @property
    def layers(self):
        return self.model.layers

    def __call__(
        self,
        input_ids,
        *,
        cache=None,
        inputs_embeds=None,
        input_embeddings=None,
        **kwargs,
    ):
        # mlx-lm calls this argument ``input_embeddings``; mlx-vlm language
        # models (and oMLX's VLM adapter) call it ``inputs_embeds``.
        embeddings = inputs_embeds if inputs_embeds is not None else input_embeddings
        kwargs.pop("return_hidden", None)
        return self.model_wrapper(input_ids, cache=cache, input_embeddings=embeddings)


class OptiqVLMModel(nn.Module):
    """Full-model facade consumed by ``VLMModelAdapter`` and engine helpers."""

    def __init__(self, model: nn.Module, frontend: Any, config: dict[str, Any]):
        super().__init__()
        self.language_model = OptiqLanguageModelAdapter(model)
        self.vision_frontend = frontend
        self.vision_tower = frontend.vision_tower
        self.config = _namespace(config)


def load_optiq_vlm(model_dir: str | Path) -> tuple[OptiqVLMModel, Any]:
    """Load a split OptiQ VLM and return an oMLX-compatible model/tokenizer."""
    path = str(model_dir)
    config = json.loads((Path(path) / "config.json").read_text())
    model_type = str(config.get("model_type", "")).lower()
    if model_type not in OPTIQ_VLM_BRIDGE_MODEL_TYPES:
        raise ValueError(f"OptiQ VLM bridge does not support model_type={model_type!r}")

    try:
        import optiq  # noqa: F401  # registers OptiQ's mlx-lm model modules
        from optiq.vlm import get_frontend
    except ImportError as exc:
        raise ImportError(
            f"Model type {model_type!r} requires mlx-optiq>=0.4.7. "
            "Install it with: pip install 'omlx[optiq]'"
        ) from exc

    from mlx_lm import load as mlx_lm_load

    frontend_factory = get_frontend(model_type)
    if frontend_factory is None:
        raise ValueError(
            f"mlx-optiq has no vision frontend registered for {model_type!r}"
        )

    model, tokenizer = mlx_lm_load(path)
    frontend = frontend_factory(path, model)
    return OptiqVLMModel(model, frontend, config), tokenizer
