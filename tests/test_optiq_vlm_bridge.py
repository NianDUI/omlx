# SPDX-License-Identifier: Apache-2.0
"""Tests for the optional OptiQ split-VLM bridge."""

import json
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx
from PIL import Image

from omlx.engine.vlm import VLMBatchedEngine
from omlx.integrations.optiq_vlm import (
    OptiqLanguageModelAdapter,
    load_optiq_vlm,
    needs_optiq_vlm_bridge,
)


def _write_config(tmp_path, model_type="mage_vl", *, sidecar=True):
    config = {"model_type": model_type}
    if sidecar:
        config["optiq_vision"] = {"sidecar": "optiq/optiq_vision.safetensors"}
    (tmp_path / "config.json").write_text(json.dumps(config))


def test_bridge_detection_requires_supported_type_and_sidecar(tmp_path):
    _write_config(tmp_path)
    assert needs_optiq_vlm_bridge(tmp_path) is True

    _write_config(tmp_path, sidecar=False)
    assert needs_optiq_vlm_bridge(tmp_path) is False

    _write_config(tmp_path, model_type="gemma4")
    assert needs_optiq_vlm_bridge(tmp_path) is False


def test_language_adapter_translates_inputs_embeds():
    inner = SimpleNamespace(layers=[object()])

    class FakeModel:
        args = SimpleNamespace(model_type="mage_vl")
        model_type = "mage_vl"
        model = inner

        def __init__(self):
            self.calls = []

        def __call__(self, input_ids, *, cache=None, input_embeddings=None):
            self.calls.append((input_ids, cache, input_embeddings))
            return "logits"

    model = FakeModel()
    adapter = OptiqLanguageModelAdapter(model)
    result = adapter("ids", cache="cache", inputs_embeds="embeds")

    assert result == "logits"
    assert model.calls == [("ids", "cache", "embeds")]
    assert adapter.layers == inner.layers


def test_load_optiq_vlm_uses_registered_frontend(tmp_path, monkeypatch):
    _write_config(tmp_path)
    raw_model = SimpleNamespace(
        args=SimpleNamespace(model_type="mage_vl"),
        model_type="mage_vl",
        model=SimpleNamespace(layers=[]),
    )
    tokenizer = object()
    frontend = SimpleNamespace(vision_tower=object())
    factory = MagicMock(return_value=frontend)

    monkeypatch.setitem(sys.modules, "optiq", types.ModuleType("optiq"))
    vlm_module = types.ModuleType("optiq.vlm")
    vlm_module.get_frontend = MagicMock(return_value=factory)
    monkeypatch.setitem(sys.modules, "optiq.vlm", vlm_module)
    mlx_lm_module = types.ModuleType("mlx_lm")
    mlx_lm_module.load = MagicMock(return_value=(raw_model, tokenizer))
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm_module)

    model, loaded_tokenizer = load_optiq_vlm(tmp_path)

    assert loaded_tokenizer is tokenizer
    assert model.config.model_type == "mage_vl"
    assert model.vision_frontend is frontend
    factory.assert_called_once_with(str(tmp_path), raw_model)
    mlx_lm_module.load.assert_called_once_with(str(tmp_path))


def test_engine_prepares_optiq_embeddings_and_forwards_tools():
    template_calls = []

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            template_calls.append((messages, kwargs))
            return [1, 2, 3]

    class FakeFrontend:
        def preprocess(self, messages, *, tokenizer, enable_thinking):
            image_part = messages[0]["content"][0]
            assert image_part["type"] == "image"
            assert isinstance(image_part["image"], Image.Image)
            token_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
            return {"input_ids": mx.array([token_ids]), "pixel_values": "pixels"}

        def merged_embeddings(self, inputs):
            return mx.zeros((1, 3, 4)), {"custom": mx.array([1])}

    engine = VLMBatchedEngine.__new__(VLMBatchedEngine)
    engine._optiq_vision_frontend = FakeFrontend()
    engine._processor = FakeTokenizer()
    engine._enable_thinking = False
    engine._vlm_model = SimpleNamespace(config=SimpleNamespace(model_type="mage_vl"))

    image = Image.new("RGB", (2, 2), "white")
    result = engine._prepare_optiq_vision_inputs(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,decoded-by-omlx",
                    },
                    {"type": "text", "text": "describe"},
                ],
            }
        ],
        [image],
        audio=None,
        tools=[{"type": "function", "function": {"name": "lookup"}}],
        is_partial=False,
    )

    token_ids, embeddings, extra, image_hash, cache_start, cache_ranges = result
    assert token_ids == [1, 2, 3]
    assert embeddings.shape == (1, 3, 4)
    assert list(extra) == ["custom"]
    assert image_hash is not None
    assert cache_start == 0
    assert cache_ranges == []
    assert template_calls[0][1]["tools"][0]["function"]["name"] == "lookup"
