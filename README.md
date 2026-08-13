<h1>
  <img src="docs/assets/bmo-mark.svg" alt="BMO icon" width="42" />
  BMO
</h1>

BMO is a modular multi-service AI backend built with Python and FastAPI, exposing computer vision (CV), LLM, speech-to-text (STT), and text-to-speech (TTS) capabilities behind a single app, each as its own router.

## Local AI BMO inspiration

<a href="https://www.youtube.com/watch?v=l5ggH-YhuAw">
  <img src="https://i.ytimg.com/vi/l5ggH-YhuAw/maxresdefault.jpg" alt="Building a Real BMO — Embodied Local AI Agent video cover" width="720" />
</a>

This project shares the same local-AI spirit as [Building a Real BMO — Embodied Local AI Agent](https://www.youtube.com/watch?v=l5ggH-YhuAw): a Raspberry Pi-powered, voice-enabled BMO using local models. The cover above links directly to the build video.

## Status

The TTS module is the most developed piece: a local PocketTTS pipeline with voice-cloning samples for small/medium/large quality modes, usable via CLI (`speak.py`), as a Python module (`TTS.cli`), or over HTTP. CV, LLM, and STT each have a scaffolded router (`core.py`), but their depth of implementation beyond that hasn't been fully verified. There's no automated test suite yet, so functional correctness of those modules is unconfirmed.

A TTS/PocketTTS rework is currently in progress in the working tree (not part of this commit).

## Quick Start Commands

Run everything from project root with plain `python`:

```bash
python main.py
python speak.py -i
python train.py --list
python exercises.py --list
python challenges.py --list
```

## Notebook Paths

Use this in the first notebook cell so reads/saves always use correct project paths:

```python
from notebook_paths import setup_notebook

paths = setup_notebook(chdir_to_root=True)
# Read file: paths.read("Assets/my_input.csv")
# Save file: paths.save("Assets/notebooks/my_output.csv")
```

## Local PocketTTS Pipeline

This project now uses a local PocketTTS speech pipeline for BMO.

The pipeline is available through:

- `POST /api/v1/tts/synthesize`
- `python speak.py`
- `python -m TTS.cli`

### CLI usage

```bash
python speak.py
python speak.py "hello world"
python speak.py input.txt
python speak.py -s "fast mode"
python speak.py -m "balanced"
python speak.py -l "best quality"
python speak.py --voice 4 "use sample 4"
```

### Modes and Voice Samples

- `small` -> `TTS/assets/voice/1.wav`
- `medium` -> `TTS/assets/voice/2.wav`
- `large` -> `TTS/assets/voice/3.wav`

You can also pick any file from `TTS/assets/voice` with `--voice` (number, stem, or filename).

## Setup

```bash
conda create -n bmo python=3.11 -y
conda activate bmo

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install "huggingface_hub[cli]"
huggingface-cli login
```

PocketTTS clip-based voice cloning requires accepting terms at `https://huggingface.co/kyutai/pocket-tts` and logging in locally (`huggingface-cli login`).
First run also downloads model weights, so keep internet available once during warmup.

## Relevant environment variables

Configure these in `.env`:

```env
TTS_PROVIDER=local
TTS_MODEL_NAME=pocket-tts
TTS_OUTPUT_FORMAT=wav
TTS_LOCAL_ENABLED=true
TTS_LOCAL_DEFAULT_MODE=small
TTS_LOCAL_OUT_DIR=TTS/assets/out
TTS_POCKET_MODEL_CONFIG=b6369a24
TTS_POCKET_VOICE_DIR=TTS/assets/voice
TTS_POCKET_CLIP_SMALL_PATH=TTS/assets/voice/1.wav
TTS_POCKET_CLIP_MEDIUM_PATH=TTS/assets/voice/2.wav
TTS_POCKET_CLIP_LARGE_PATH=TTS/assets/voice/3.wav
```

## API example

```json
{
  "text": "BMO says hello",
  "mode": "medium",
  "voice": "4"
}
```
