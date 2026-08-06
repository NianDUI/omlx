# SPDX-License-Identifier: Apache-2.0
"""Tests for the OpenAI-compatible image generation endpoint."""

import base64
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from omlx.engine.image_generation import GeneratedImageOutput


class _FakeImageEngine:
    def __init__(self):
        self.generate = AsyncMock(
            return_value=GeneratedImageOutput(png_bytes=b"fake-png")
        )


def test_image_generation_returns_base64_png():
    from omlx.server import app

    engine = _FakeImageEngine()

    @asynccontextmanager
    async def acquire(_model):
        yield engine

    with (
        patch("omlx.server.acquire_image_generation_engine", acquire),
        TestClient(app, raise_server_exceptions=False) as client,
    ):
        response = client.post(
            "/v1/images/generations",
            json={
                "model": "flux2-klein-4b-4bit",
                "prompt": "a red apple",
                "size": "512x768",
                "steps": 4,
                "seed": 42,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert base64.b64decode(body["data"][0]["b64_json"]) == b"fake-png"
    engine.generate.assert_awaited_once_with(
        "a red apple",
        seed=42,
        steps=4,
        width=512,
        height=768,
        guidance=1.0,
    )


def test_image_generation_rejects_one_step():
    from omlx.server import app

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/v1/images/generations",
            json={
                "model": "flux2-klein-4b-4bit",
                "prompt": "a red apple",
                "steps": 1,
            },
        )

    assert response.status_code == 422


def test_image_generation_rejects_non_multiple_of_16_size():
    from omlx.server import app

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/v1/images/generations",
            json={
                "model": "flux2-klein-4b-4bit",
                "prompt": "a red apple",
                "size": "513x512",
            },
        )

    assert response.status_code == 422


def test_discovery_registers_mflux_model(tmp_path):
    from omlx.model_discovery import discover_models

    model = tmp_path / "flux2-klein-4b-4bit"
    model.mkdir()
    (model / "configuration.json").write_text(
        '{"framework":"pytorch","task":"text-to-image"}'
    )
    transformer = model / "transformer"
    transformer.mkdir()
    (transformer / "0.safetensors").write_bytes(b"weights")

    models = discover_models(tmp_path)

    assert models[model.name].model_type == "image_generation"
    assert models[model.name].engine_type == "image_generation"
