import io
import json
from pathlib import Path

from .characters import policy, voice_metadata


def manifest_voices(manifest: dict) -> list[dict]:
    voices = []
    for filename, characters in manifest.items():
        for character in characters:
            name = character["name"]
            if not policy(name).ja_allowed:
                continue
            for style in character["styles"]:
                if style.get("type", "talk") != "talk":
                    continue
                voices.append({"id": f"ja:{style['id']}", "name": f"{name} · {style['name']}",
                               "kind": "ja", "language": "ja", "region": None,
                               "description": "Tiếng Nhật", **voice_metadata(name)})
    return voices


class VoicevoxEngine:
    def __init__(self, models: Path):
        self.directory = models / "voicevox"
        self.loaded = set()
        self.voices = []

    def load(self):
        from voicevox_core.blocking import Onnxruntime, OpenJtalk, Synthesizer

        runtime = self.directory / "voicevox_onnxruntime-osx-arm64-1.17.3/lib/libvoicevox_onnxruntime.1.17.3.dylib"
        ort = Onnxruntime.load_once(filename=str(runtime))
        ojt = OpenJtalk(str(self.directory / "open_jtalk_dic_utf_8-1.11"))
        self.synthesizer = Synthesizer(ort, ojt)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        self.voices = manifest_voices(self.manifest)
        self.style_models = {f"ja:{style['id']}": filename
                             for filename, characters in self.manifest.items()
                             for character in characters for style in character["styles"]}
        return self

    def synthesize(self, text, voice_id, speed, pitch):
        import soundfile as sf
        from voicevox_core.blocking import VoiceModelFile

        filename = self.style_models[voice_id]
        if filename not in self.loaded:
            with VoiceModelFile.open(self.directory / filename) as model:
                self.synthesizer.load_voice_model(model)
            self.loaded.add(filename)
        style_id = int(voice_id.split(":", 1)[1])
        query = self.synthesizer.create_audio_query(text, style_id)
        query.speed_scale = speed
        # VOICEVOX multiplies frequency by 2**pitch_scale; semitones are 1/12 octave.
        query.pitch_scale = pitch / 12.0
        wav = self.synthesizer.synthesis(query, style_id)
        audio, rate = sf.read(io.BytesIO(wav), dtype="float32", always_2d=False)
        return audio, rate
