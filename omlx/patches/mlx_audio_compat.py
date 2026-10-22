# SPDX-License-Identifier: Apache-2.0
"""Compatibility helpers for mlx-audio dependency drift."""

from __future__ import annotations

import json
import logging
from functools import wraps
from pathlib import Path

logger = logging.getLogger(__name__)

_RESAMPLE_EXPORT_CHECKED = False


def ensure_vibevoice_asr_tokenizer() -> bool:
    """Select Qwen2 tokenization for legacy VibeVoice ASR checkpoints.

    Transformers 5 validates their shared ``vibevoice`` model type as TTS,
    ignoring ``decoder_config`` and rejecting the hidden-size mismatch. The
    mlx-audio hook then falls back to a base tokenizer without ASR tokens.
    Explicit tokenizer selection skips that unrelated model-config validation.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return False

    original = AutoTokenizer.from_pretrained
    if getattr(original, "_omlx_vibevoice_asr", False) is True:
        return True

    @wraps(original)
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        if "tokenizer_type" not in kwargs and "config" not in kwargs:
            try:
                config_path = Path(pretrained_model_name_or_path) / "config.json"
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                config = {}
            if config.get("model_type") == "vibevoice" and any(
                arch in {"VibeVoiceForASRTraining", "VibeVoiceForASRStreamingTraining"}
                for arch in config.get("architectures", [])
            ):
                kwargs["tokenizer_type"] = "qwen2"
        return original(pretrained_model_name_or_path, *inputs, **kwargs)

    from_pretrained._omlx_vibevoice_asr = True
    AutoTokenizer.from_pretrained = classmethod(from_pretrained)
    return True


def ensure_qwen3_asr_audio_quantization() -> bool:
    """Let the loader check Qwen3-ASR audio scales and per-layer settings."""
    try:
        from mlx_audio.stt.models.qwen3_asr.qwen3_asr import Qwen3ASRModel
    except ImportError:
        return False

    original = Qwen3ASRModel.model_quant_predicate
    if getattr(original, "_omlx_audio_quantization", False):
        return True

    @wraps(original)
    def model_quant_predicate(self, path, module):
        # The shared loader still checks scales or an explicit layer override.
        if path.startswith("audio_tower."):
            return True
        return original(self, path, module)

    model_quant_predicate._omlx_audio_quantization = True
    Qwen3ASRModel.model_quant_predicate = model_quant_predicate
    return True


def ensure_mlx_audio_resample_export() -> bool:
    """Ensure mlx-vlm's legacy resampler import path is available.

    mlx-vlm@526c210 imports ``resample_audio`` from ``mlx_audio.utils`` inside
    ``load_audio()``.  mlx-audio@5175326 moved that function to
    ``mlx_audio.stt.utils`` without re-exporting it from the old location.
    """
    global _RESAMPLE_EXPORT_CHECKED

    try:
        import mlx_audio.utils as audio_utils
    except ImportError:
        return False

    if hasattr(audio_utils, "resample_audio"):
        _RESAMPLE_EXPORT_CHECKED = True
        return True

    try:
        from mlx_audio.stt.utils import resample_audio
    except ImportError:
        return False

    audio_utils.resample_audio = resample_audio
    if not _RESAMPLE_EXPORT_CHECKED:
        logger.debug("mlx_audio.utils.resample_audio compatibility export applied")
    _RESAMPLE_EXPORT_CHECKED = True
    return True
