from pathlib import Path

from .characters import clone_characters, voice_metadata
from .models import slugify


def patch_fetch(models: Path) -> None:
    # Real local files avoid ONNX external-data escaping through HF blob symlinks.
    from vieneu._v3_turbo_engine.onnx_runtime_lite import OnnxV3LiteEngine

    def fetch(repo, files, subfolder):
        from huggingface_hub import hf_hub_download

        local = models / "vieneu" / repo.replace("/", "--")
        directory = local / subfolder if subfolder else local
        directory.mkdir(parents=True, exist_ok=True)
        for filename in files:
            target = directory / filename
            if not target.is_file():
                hf_hub_download(repo, filename, subfolder=subfolder or None, local_dir=local)
            if not target.is_file() or target.is_symlink():
                raise RuntimeError(f"TTS asset is not a real file: {target}; run setup")
        return directory

    OnnxV3LiteEngine._fetch = staticmethod(fetch)


def preset_voice(label: str) -> dict:
    name, _, description = label.lstrip("⭐ ").partition("—")
    parts = [part.strip() for part in description.split("·")]
    return {"id": f"vi:{slugify(name.strip())}", "name": name.strip(), "kind": "vi",
            "language": "vi", "gender": "female" if parts[0] == "Nữ" else
            "male" if parts[0] == "Nam" else "unknown",
            "region": parts[1] if len(parts) > 1 else None,
            "description": " · ".join(parts[2:]) or None,
            "credit": None, "terms_url": None}


class VieneuEngine:
    def __init__(self, models: Path):
        self.models = models
        self.voices = []
        self.voice_names = {}

    def load(self):
        patch_fetch(self.models)
        from vieneu import Vieneu

        self.tts = Vieneu(device="cpu", backend="onnx")
        for label, identity in self.tts.list_preset_voices():
            voice = preset_voice(label)
            self.voices.append(voice)
            self.voice_names[voice["id"]] = identity
        # Re-enrol references rather than writing into the installed package's voices file.
        for name, item in clone_characters():
            reference = self.models / "refs" / f"{item.slug}.wav"
            if reference.is_file():
                identity = f"vi-anime:{item.slug}"
                self.tts.add_voice(identity, reference, denoise=False)
                self.voice_names[identity] = identity
                self.voices.append({"id": identity, "name": f"{item.slug.replace('-', ' ').title()} ({name})",
                                    "kind": "vi-anime", "language": "vi", "region": None,
                                    "description": "Giọng anime, nói tiếng Việt", **voice_metadata(name)})
        return self

    def synthesize(self, text, voice_id, speed, pitch):
        import numpy as np

        audio = np.asarray(self.tts.infer(text, voice=self.voice_names[voice_id]), dtype=np.float32)
        if pitch != 0 or speed != 1:
            import librosa

            if pitch != 0:
                audio = librosa.effects.pitch_shift(audio, sr=48000, n_steps=pitch)
            if speed != 1:
                audio = librosa.effects.time_stretch(audio, rate=speed)
        return audio, 48000
