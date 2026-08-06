# SPDX-License-Identifier: Apache-2.0
"""Text-to-image engine backed by mflux on Apple Silicon."""

import asyncio
import gc
import io
import logging
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from ..engine_core import get_mlx_executor
from .base import BaseNonStreamingEngine

logger = logging.getLogger(__name__)


@dataclass
class GeneratedImageOutput:
    """PNG output and generation metadata."""

    png_bytes: bytes
    generation_time: float | None = None


class ImageGenerationEngine(BaseNonStreamingEngine):
    """Non-streaming mflux engine for FLUX.2 Klein image generation."""

    def __init__(self, model_name: str, **kwargs):
        super().__init__()
        self._model_name = model_name
        self._model = None
        self._kwargs = kwargs
        self._generation_lock = asyncio.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    async def start(self) -> None:
        if self._model is not None:
            return

        try:
            from mflux.models.common.config import ModelConfig
            from mflux.models.flux2.variants import Flux2Klein
        except ImportError as exc:
            raise ImportError(
                "mflux is required for image generation. "
                'Install dependencies with `pip install "omlx[image]"`, then '
                "install mflux with `pip install mflux==0.18.0 --no-deps`."
            ) from exc

        model_name = self._model_name

        def _load_sync():
            config = ModelConfig.from_name(model_name=model_name)
            return Flux2Klein(model_config=config, **self._kwargs)

        logger.info("Starting image generation engine: %s", model_name)
        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(get_mlx_executor(), _load_sync)
        logger.info("Image generation engine started: %s", model_name)

    async def stop(self) -> None:
        if self._model is None:
            return
        self._model = None
        gc.collect()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            get_mlx_executor(), lambda: (mx.synchronize(), mx.clear_cache())
        )

    async def generate(
        self,
        prompt: str,
        *,
        seed: int,
        steps: int = 4,
        width: int = 1024,
        height: int = 1024,
        guidance: float = 1.0,
    ) -> GeneratedImageOutput:
        if self._model is None:
            raise RuntimeError("Engine not started. Call start() first.")

        model = self._model

        def _generate_sync() -> GeneratedImageOutput:
            generated = model.generate_image(
                seed=seed,
                prompt=prompt,
                num_inference_steps=steps,
                width=width,
                height=height,
                guidance=guidance,
            )
            buffer = io.BytesIO()
            generated.image.save(buffer, format="PNG")
            return GeneratedImageOutput(
                png_bytes=buffer.getvalue(),
                generation_time=getattr(generated, "generation_time", None),
            )

        activity_id = self._begin_activity(
            "generating image",
            detail="Generating image",
            metadata={"width": width, "height": height, "steps": steps},
        )
        try:
            async with self._generation_lock:
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(get_mlx_executor(), _generate_sync)
        finally:
            await self._finish_activity(activity_id)

    def get_stats(self) -> dict[str, Any]:
        return {
            "model_name": self._model_name,
            "loaded": self._model is not None,
        }
