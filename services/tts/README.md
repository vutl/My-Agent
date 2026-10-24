# Aya local speech

Optional CPU sidecar at `127.0.0.1:7801`. The desktop calls only the backend's
`/speech/*` API on :7777. VieNeu v3 Turbo speaks Vietnamese (presets and approved
anime clones); VOICEVOX speaks Japanese after the backend translates the cleaned
answer using the configured router model. Chat and vision model settings are unchanged.

## Install once (Apple Silicon macOS; network required)

From the repository root, use Python 3.12 for the separate environment:

```sh
uv venv --python 3.12 services/tts/.venv
uv pip install --python services/tts/.venv/bin/python -e services/tts
uv pip install --python services/tts/.venv/bin/python 'https://github.com/VOICEVOX/voicevox_core/releases/download/0.17.1/voicevox_core-0.17.1-cp310-abi3-macosx_11_0_arm64.whl'
services/tts/.venv/bin/python services/tts/setup_tts.py
```

The setup downloads the VieNeu ONNX graphs, codec and clone assets, VOICEVOX
0.17.0 VVMs (0–24, n0, s0), the VOICEVOX 1.17.3 runtime and Open JTalk 1.11
dictionary. It writes `voicevox/manifest.json` and 5–8 second references under
`refs/`. Existing files are skipped; incomplete downloads use `.part` files.
Reference enrollment warms the speaker encoder and denoiser downloads. Profiles
are re-enrolled from the reference WAVs on startup, without modifying the
installed package's preset file.

Default models directory: `<repo>/data/models/tts`. Use
`setup_tts.py --models-dir /absolute/path` or `AYA_TTS_MODELS` to override it.
Set `TTS_MODELS_DIR` to the same path in the backend environment.
The ONNX `_fetch` patch uses real files under `vieneu/<org>--<repo>/`, avoiding
ONNX external-data failures with Hugging Face blob symlinks. Auxiliary HF assets
live in `<models>/huggingface`, shared by setup and runtime.

Restart the backend after setup. It starts the sidecar in its lifespan and logs
to `<APP_DATA_DIR>/logs/tts-sidecar.log`. An existing healthy port is reused and
that borrowed process is never terminated by the backend. Owned processes get
up to three quick restarts with exponential backoff; a 60 second stable ready
period resets that budget. A borrowed sidecar's failure is reported as `error`
so its owner can repair it. Startup is bounded to 120 seconds per attempt.
Runtime sets `HF_HUB_OFFLINE=1` and disables telemetry before engine imports;
missing assets produce an error instead of downloading. To run manually:

```sh
cd services/tts
AYA_TTS_MODELS=/absolute/path/to/models .venv/bin/python -m aya_tts.server --port 7801
```

## Backend configuration and contract

| Environment variable | Default |
| --- | --- |
| `TTS_ENABLED` | `true` |
| `TTS_PORT` | `7801` |
| `TTS_PYTHON` | `<repo>/services/tts/.venv/bin/python` |
| `TTS_MODELS_DIR` | `<repo>/data/models/tts` |
| `TTS_DEFAULT_VOICE` | `kasukabe-tsumugi` (slug or full `vi-anime:` id) |

- `GET /speech/status`: `off`, `starting`, `ready` or `error`, with `detail`.
  Disabled or uninstalled TTS leaves the rest of Aya available.
- `GET /speech/voices`: voice metadata and `default_voice_id`; 503 until ready.
  Preferred anime voice falls back to the first Vietnamese female preset.
- `POST /speech/synthesize`: `text`, `voice_id`, `speed` (0.7–1.5, default 1),
  `pitch` (−4…+6 semitones, default 0). Returns mono 16-bit PCM `audio/wav`.
  `X-Aya-Spoken-Text` contains percent-encoded UTF-8 and is exposed through CORS.
  204 means no speech remains; 404 unknown voice; 422 invalid parameters;
  503 unavailable; 502 translation/engine failure. Decode the header before display.
- Code, LaTeX, tables, citation markers, URLs and Markdown decoration are removed;
  link text and Vietnamese diacritics are preserved. Japanese translation has a
  20 second deadline and a 128-entry LRU cache. No model fallback is introduced.
- Internal sidecar endpoints: `/health` (also `load_seconds`), `/voices` (no
  default), `/synthesize` (already cleaned/translated text). Engines initialize
  in a background thread, VOICEVOX VVMs load lazily, each engine serializes its
  own synthesis calls. WAV encoding uses stdlib `wave` and NumPy, so the backend
  test environment does not need `soundfile`; model audio processing still uses it.

## Credits and character terms

The policy registry is `aya_tts/characters.py`. Every VOICEVOX-derived voice
carries its exact credit and terms URL; clients must show them. Japanese output
is permitted for unknown characters with `VOICEVOX:<name>` credit; unknown or
unverified characters are never cloned. Male characters are Japanese-only.
The cloning policy follows the approved personal-use contract; distribution
and other uses must comply with each linked character's terms.

VieNeu: [Apache-2.0 model](https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Turbo).
VOICEVOX: [core releases](https://github.com/VOICEVOX/voicevox_core/releases/tag/0.17.1),
[models](https://github.com/VOICEVOX/voicevox_vvm/releases/tag/0.17.0).

| Character | Credit | Vietnamese clone | Terms |
| --- | --- | --- | --- |
| 四国めたん | VOICEVOX:四国めたん | Yes | [Terms](https://zunko.jp/con_ongen_kiyaku.html) |
| ずんだもん | VOICEVOX:ずんだもん | Yes | [Terms](https://zunko.jp/con_ongen_kiyaku.html) |
| 春日部つむぎ | VOICEVOX:春日部つむぎ | Yes | [Terms](https://tsumugi-official.studio.site/rule) |
| 九州そら | VOICEVOX:九州そら | Yes | [Terms](https://zunko.jp/con_ongen_kiyaku.html) |
| 冥鳴ひまり | VOICEVOX:冥鳴ひまり | Yes | [Terms](https://meimeihimari.wixsite.com/himari/terms-of-use) |
| 波音リツ | VOICEVOX:波音リツ | Yes | [Terms](https://www.canon-voice.com/kiyaku.html) |
| 春歌ナナ | VOICEVOX:春歌ナナ | Yes | [Terms](https://confetto.chu.jp/haruka/) |
| 小夜/SAYO | VOICEVOX:小夜/SAYO | Yes | [Terms](https://316soramegu.wixsite.com/sayo/guide) |
| 中国うさぎ | VOICEVOX:中国うさぎ | Yes | [Terms](https://zunko.jp/con_ongen_kiyaku.html) |
| 東北ずん子 | VOICEVOX:東北ずん子 | Yes | [Terms](https://zunko.jp/con_ongen_kiyaku.html) |
| 東北きりたん | VOICEVOX:東北きりたん | Yes | [Terms](https://zunko.jp/con_ongen_kiyaku.html) |
| 東北イタコ | VOICEVOX:東北イタコ | Yes | [Terms](https://zunko.jp/con_ongen_kiyaku.html) |
| 櫻歌ミコ | VOICEVOX:櫻歌ミコ | Yes | [Terms](https://miko35.info/) |
| 雨晴はう | VOICEVOX:雨晴はう | No | [Terms](https://amehau.com/) |
| 猫使ビィ | VOICEVOX:猫使ビィ | No | [Terms](https://nekonohako.fanbox.cc/posts/4624073) |
| 猫使アル | VOICEVOX:猫使アル | No | [Terms](https://nekonohako.fanbox.cc/posts/4624073) |
| もち子さん | VOICEVOX:もち子(cv 明日葉よもぎ) | No | [Terms](https://vtubermochio.wixsite.com/mochizora/利用規約) |
| 栗田まろん | VOICEVOX:栗田まろん | No | [Terms](https://aivoice.jp/character/maron/) |
| 琴詠ニア | VOICEVOX:琴詠ニア | No | [Terms](https://commons.nicovideo.jp/works/nc315435) |
| 玄野武宏 | VOICEVOX:玄野武宏 | No | [Terms](https://virvoxproject.wixsite.com/official/) |
| 白上虎太郎 | VOICEVOX:白上虎太郎 | No | [Terms](https://virvoxproject.wixsite.com/official/) |
| 青山龍星 | VOICEVOX:青山龍星 | No | [Terms](https://virvoxproject.wixsite.com/official/) |
| 剣崎雌雄 | VOICEVOX:剣崎雌雄 | No | [Terms](https://virvoxproject.wixsite.com/official/) |

## Offline tests

From `backend/` (or use the backend venv's absolute path in a worktree):

```sh
PYTHONPATH=.. .venv/bin/python -m pytest -q -p no:cacheprovider
PYTHONPATH=..:../services/tts .venv/bin/python -m pytest -q -p no:cacheprovider ../services/tts/tests
```

Tests replace engines, subprocess creation and HTTP with fakes; no models or
network are used. Importing `app.main` or `aya_tts.server` does not start a
sidecar or initialize either engine. Run online setup and real audio listening
checks separately after installation; fake-engine tests do not assess voice quality.
