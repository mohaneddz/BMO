# TTS

Local PocketTTS voice synthesis for BMO.

Pipeline:

`text -> PocketTTS(audio-conditioned on clip) -> wav`

This module supports:

- local CLI generation
- local FastAPI generation through `/api/v1/tts/synthesize`

## Files

- `cli.py`: CLI entrypoint logic
- `core.py`: FastAPI router
- `local_pipeline.py`: shared PocketTTS pipeline
- `assets/voice/`: voice sample clips used for conditioning
- `assets/out/`: generated wav files

## Three-clip setup

The mode chooses the clip:

- `small` -> `TTS/assets/voice/1.wav`
- `medium` -> `TTS/assets/voice/2.wav`
- `large` -> `TTS/assets/voice/3.wav`

You can also pick any clip in `TTS/assets/voice` by passing `--voice` in CLI or `voice` in API.

## Installation

From repo root:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install "huggingface_hub[cli]"
huggingface-cli login
```

For clip-based voice cloning, accept terms at `https://huggingface.co/kyutai/pocket-tts` and login with `huggingface-cli login`.
PocketTTS downloads model weights on first run.

## CLI

```powershell
python speak.py "Hello world"
python speak.py -s "Use small clip"
python speak.py -m "Use medium clip"
python speak.py -l "Use large clip"
python speak.py --voice 4 "Use sample 4.wav"
python speak.py -i
python speak.py .\my_text.txt
```

Interactive mode (`-i`) plays audio from memory and does not save per-line files.

## Output

Final wav files are saved to:

`TTS/assets/out`

## API

Route:

`POST /api/v1/tts/synthesize`

Example body:

```json
{
  "text": "BMO says hello",
  "mode": "large",
  "voice": "4"
}
```

Response includes:

- `file_path`
- `mode`
- `pipeline`
- `voice`
- `model`
- `timings_seconds`

## Environment

Relevant `.env` values:

```env
ASSETS_TTS_DIR=TTS/assets
TTS_LOCAL_OUT_DIR=TTS/assets/out
TTS_LOCAL_USE_WARM_ENGINE=true
TTS_MODEL_NAME=pocket-tts
TTS_OUTPUT_FORMAT=wav
TTS_POCKET_MODEL_CONFIG=b6369a24
TTS_POCKET_TEMPERATURE=0.7
TTS_POCKET_LSD_DECODE_STEPS=1
TTS_POCKET_EOS_THRESHOLD=-4.0
TTS_POCKET_MAX_TOKENS=1200
TTS_POCKET_FRAMES_AFTER_EOS=4
TTS_POCKET_VOICE_DIR=TTS/assets/voice
TTS_POCKET_CLIP_SMALL_PATH=TTS/assets/voice/1.wav
TTS_POCKET_CLIP_MEDIUM_PATH=TTS/assets/voice/2.wav
TTS_POCKET_CLIP_LARGE_PATH=TTS/assets/voice/3.wav
```
