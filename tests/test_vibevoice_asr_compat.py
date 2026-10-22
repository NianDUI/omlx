# SPDX-License-Identifier: Apache-2.0
"""Legacy VibeVoice ASR tokenizers must retain their checkpoint tokens."""

import json
from unittest.mock import Mock

import pytest
from transformers import AutoTokenizer

from omlx.patches.mlx_audio_compat import ensure_vibevoice_asr_tokenizer


@pytest.mark.parametrize("architecture", [
    "VibeVoiceForASRTraining",
    "VibeVoiceForASRStreamingTraining",
    "VibeVoiceForConditionalGeneration",
    "VibeVoiceStreamingForConditionalGenerationInference",
])
def test_tokenizer_selection(tmp_path, monkeypatch, architecture):
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "vibevoice", "architectures": [architecture],
    }))
    original = Mock()
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", original)
    assert ensure_vibevoice_asr_tokenizer()
    patched = AutoTokenizer.from_pretrained
    assert ensure_vibevoice_asr_tokenizer()
    assert AutoTokenizer.from_pretrained == patched

    AutoTokenizer.from_pretrained(str(tmp_path), local_files_only=True)
    expected = {"local_files_only": True}
    if "ForASR" in architecture:
        expected["tokenizer_type"] = "qwen2"
    original.assert_called_once_with(str(tmp_path), **expected)


@pytest.mark.parametrize("kwargs", [
    {"tokenizer_type": "custom"}, {"config": object()}, {},
])
def test_other_tokenizer_calls_unchanged(tmp_path, monkeypatch, kwargs):
    original = Mock()
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", original)
    ensure_vibevoice_asr_tokenizer()
    AutoTokenizer.from_pretrained(str(tmp_path), **kwargs)
    original.assert_called_once_with(str(tmp_path), **kwargs)
