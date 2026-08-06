# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible image generation request and response models."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ImageGenerationRequest(BaseModel):
    prompt: str = Field(min_length=1)
    model: str
    n: int = Field(default=1, ge=1, le=1)
    quality: str | None = None
    response_format: Literal["b64_json"] = "b64_json"
    size: str = "1024x1024"
    user: str | None = None
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    steps: int = Field(default=4, ge=2, le=50)
    guidance: float = Field(default=1.0, ge=0.0, le=20.0)

    @field_validator("size")
    @classmethod
    def validate_size(cls, value: str) -> str:
        try:
            width_text, height_text = value.lower().split("x", 1)
            width, height = int(width_text), int(height_text)
        except (TypeError, ValueError) as exc:
            raise ValueError("size must use WIDTHxHEIGHT format") from exc
        if not (256 <= width <= 2048 and 256 <= height <= 2048):
            raise ValueError("image dimensions must be between 256 and 2048")
        if width % 16 or height % 16:
            raise ValueError("image dimensions must be multiples of 16")
        return f"{width}x{height}"

    def dimensions(self) -> tuple[int, int]:
        width, height = self.size.split("x", 1)
        return int(width), int(height)


class ImageData(BaseModel):
    b64_json: str
    revised_prompt: str | None = None


class ImageGenerationResponse(BaseModel):
    created: int
    data: list[ImageData]
