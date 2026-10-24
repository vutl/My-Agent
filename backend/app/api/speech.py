from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field
import httpx

from app.core.config import get_settings
from app.services.speech import speech_text, translate_ja

router = APIRouter(prefix="/speech", tags=["speech"])


class SynthesisRequest(BaseModel):
    text: str
    voice_id: str = Field(min_length=1)
    speed: float = Field(default=1.0, ge=0.7, le=1.5, allow_inf_nan=False)
    pitch: float = Field(default=0.0, ge=-4.0, le=6.0, allow_inf_nan=False)


def get_supervisor(request: Request):
    return getattr(request.app.state, "speech_supervisor", None)


def speech_status(supervisor) -> dict:
    return supervisor.status() if supervisor else {"state": "off", "detail": None}


@router.get("/status")
async def status(supervisor=Depends(get_supervisor)):
    return speech_status(supervisor)


async def sidecar_request(supervisor, method, path, **kwargs):
    if speech_status(supervisor)["state"] != "ready":
        raise HTTPException(503, "Speech is not ready")
    try:
        response = await supervisor.client.request(
            method, f"{supervisor.base_url}{path}", timeout=60.0, **kwargs,
        )
    except httpx.HTTPError as exc:
        raise HTTPException(503, "Speech sidecar is unavailable") from exc
    if response.status_code in {404, 422, 503}:
        raise HTTPException(response.status_code, "Speech sidecar rejected the request")
    if response.status_code != 200:
        raise HTTPException(502, "Speech sidecar failed")
    return response


async def voice_list(supervisor):
    response = await sidecar_request(supervisor, "GET", "/voices")
    try:
        voices = response.json()["voices"]
        if not isinstance(voices, list) or any(not isinstance(voice, dict) or "id" not in voice for voice in voices):
            raise ValueError("Invalid voices")
        return voices
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(502, "Invalid sidecar voice list") from exc


@router.get("/voices")
async def voices(supervisor=Depends(get_supervisor)):
    items = await voice_list(supervisor)
    preferred = get_settings().tts_default_voice.removeprefix("vi-anime:")
    default = next((voice["id"] for voice in items
                    if voice.get("kind") == "vi-anime" and voice["id"] == f"vi-anime:{preferred}"), None)
    if default is None:
        default = next((voice["id"] for voice in items
                        if voice.get("kind") == "vi" and voice.get("gender") == "female"), None)
    return {"voices": items, "default_voice_id": default}


@router.post("/synthesize")
async def synthesize(body: SynthesisRequest, supervisor=Depends(get_supervisor)):
    items = await voice_list(supervisor)
    voice = next((voice for voice in items if voice["id"] == body.voice_id), None)
    if voice is None:
        raise HTTPException(404, "Unknown voice")
    spoken = speech_text(body.text)
    if not spoken:
        return Response(status_code=204)
    if voice.get("kind") == "ja":
        try:
            spoken = await translate_ja(spoken)
        except Exception as exc:
            raise HTTPException(502, "Japanese speech translation failed") from exc
    response = await sidecar_request(supervisor, "POST", "/synthesize", json={
        "text": spoken, "voice_id": body.voice_id, "speed": body.speed, "pitch": body.pitch,
    })
    return Response(response.content, media_type="audio/wav",
                    headers={"X-Aya-Spoken-Text": quote(spoken, safe="")})
