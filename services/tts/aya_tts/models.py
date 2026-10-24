import re
import unicodedata

from pydantic import BaseModel, Field, field_validator


def slugify(name: str) -> str:
    plain = unicodedata.normalize("NFKD", name.casefold().replace("đ", "d"))
    return re.sub(r"[^a-z0-9]+", "-", plain.encode("ascii", "ignore").decode()).strip("-")


def parse_voice_id(value: str) -> tuple[str, str]:
    kind, sep, identity = value.partition(":")
    valid = identity.isascii() and identity.isdigit() if kind == "ja" else bool(
        re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", identity)
    )
    if not sep or kind not in {"vi", "vi-anime", "ja"} or not valid:
        raise ValueError("Invalid voice id")
    return kind, identity


class SynthesisRequest(BaseModel):
    text: str = Field(min_length=1)
    voice_id: str
    speed: float = Field(default=1.0, ge=0.7, le=1.5, allow_inf_nan=False)
    pitch: float = Field(default=0.0, ge=-4.0, le=6.0, allow_inf_nan=False)

    @field_validator("voice_id")
    @classmethod
    def valid_voice_id(cls, value: str) -> str:
        parse_voice_id(value)
        return value
