"""One-time online asset installation. Runtime never calls this module."""
import argparse
import io
import json
import os
from pathlib import Path
import tarfile
import urllib.request

from .characters import clone_characters, policy
from .config import models_dir

VVM_RELEASE = "https://github.com/VOICEVOX/voicevox_vvm/releases/download/0.17.0"
RUNTIME_URL = ("https://github.com/VOICEVOX/onnxruntime-builder/releases/download/"
               "voicevox_onnxruntime-1.17.3/voicevox_onnxruntime-osx-arm64-1.17.3.tgz")
DICTIONARY_URL = ("https://github.com/r9y9/open_jtalk/releases/download/v1.11.1/"
                  "open_jtalk_dic_utf_8-1.11.tar.gz")
REFERENCE_TEXT = "こんにちは、今日はとてもいい天気だね。あなたとお話しできて、とってもうれしいよ。"


def download(url: str, target: Path):
    if target.is_file():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def extract(url: str, directory: Path, expected: Path):
    if expected.exists():
        return
    archive = directory / url.rsplit("/", 1)[-1]
    download(url, archive)
    with tarfile.open(archive) as tar:
        # Python 3.11+ data filter rejects traversal and unsafe symlinks.
        tar.extractall(directory, filter="data")
    if not expected.exists():
        raise RuntimeError(f"Archive did not contain {expected}")


def setup(models: Path):
    # Set before importing huggingface_hub/vieneu. Keep auxiliary HF assets in
    # the same installation as the real ONNX files, independent of user caches.
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HOME"] = str(models / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(models / "huggingface/hub")
    import numpy as np
    import soundfile as sf
    from voicevox_core.blocking import Onnxruntime, OpenJtalk, Synthesizer, VoiceModelFile

    directory = models / "voicevox"
    directory.mkdir(parents=True, exist_ok=True)
    runtime = directory / "voicevox_onnxruntime-osx-arm64-1.17.3/lib/libvoicevox_onnxruntime.1.17.3.dylib"
    dictionary = directory / "open_jtalk_dic_utf_8-1.11"
    extract(RUNTIME_URL, directory, runtime)
    extract(DICTIONARY_URL, directory, dictionary)
    synthesizer = Synthesizer(Onnxruntime.load_once(filename=str(runtime)), OpenJtalk(str(dictionary)))
    manifest = {}
    manifest_path = directory / "manifest.json"
    excluded_path = directory / "excluded-models.json"
    excluded = set(json.loads(excluded_path.read_text())) if excluded_path.is_file() else set()
    refs = models / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    clones = dict(clone_characters())
    for filename in [f"{number}.vvm" for number in range(25)] + ["n0.vvm", "s0.vvm"]:
        if filename in excluded:
            continue
        target = directory / filename
        download(f"{VVM_RELEASE}/{filename}", target)
        with VoiceModelFile.open(target) as model:
            characters = [{"name": meta.name, "styles": [
                {"id": style.id, "name": style.name, "type": getattr(style.type, "value", style.type)}
                for style in meta.styles]} for meta in model.metas]
            if not any(policy(character["name"]).ja_allowed for character in characters):
                excluded.add(filename)
                continue
            manifest[filename] = characters
            needed = [(character, clones[character["name"]]) for character in characters
                      if character["name"] in clones
                      and not (refs / f"{clones[character['name']].slug}.wav").is_file()]
            if needed:
                synthesizer.load_voice_model(model)
                for character, item in needed:
                    styles = [style for style in character["styles"] if style["type"] == "talk"]
                    if not styles:
                        continue
                    style = next((style for style in styles if style["name"] == "ノーマル"), styles[0])
                    wav = synthesizer.tts(REFERENCE_TEXT, style["id"])
                    audio, rate = sf.read(io.BytesIO(wav), dtype="float32")
                    audio = audio[:8 * rate]
                    if len(audio) < 5 * rate:
                        audio = np.pad(audio, (0, 5 * rate - len(audio)))
                    sf.write(refs / f"{item.slug}.wav", audio, rate, subtype="PCM_16")
                synthesizer.unload_voice_model(model.id)
        if filename in excluded:
            target.unlink(missing_ok=True)
    # Also prune files excluded by policy, including reruns after a policy edit.
    for filename in excluded:
        (directory / filename).unlink(missing_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    excluded_path.write_text(json.dumps(sorted(excluded)))

    from .vieneu_engine import VieneuEngine

    # Loading enrols each allowed reference and downloads speaker/denoiser assets
    # once. No profiles are written into the package's built-in voices file.
    engine = VieneuEngine(models).load()
    if not any(voice["kind"] == "vi-anime" for voice in engine.voices):
        raise RuntimeError("No clone references were prepared")
    print(f"Installed {len(manifest)} VOICEVOX models, {sum(len(c) for c in manifest.values())} characters, "
          f"{len(engine.voices)} Vietnamese voices in {models}")
    print("Setup complete. Runtime uses HF_HUB_OFFLINE=1.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=models_dir())
    args = parser.parse_args()
    setup(args.models_dir.expanduser().resolve())


if __name__ == "__main__":
    main()
