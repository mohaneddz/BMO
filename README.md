![BMO](screenshots/cover.avif)

BMO is a local AI backend in Python and FastAPI. One app (`main.py`) mounts four modules, each in its own folder with a `core.py` router:

| Module | Route | What runs | Weights |
| --- | --- | --- | --- |
| STT | `POST /api/v1/stt/transcribe` | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) `base`, CPU int8 | ~145 MB, fetched from Hugging Face on the first request |
| CV | `POST /api/v1/cv/analyze` | [Ultralytics](https://docs.ultralytics.com/) YOLO26n object detection | 5.3 MB, fetched to `CV/weights/` on the first request |
| LLM | `POST /api/v1/llm/chat` | Groq chat completions (`llama-3.3-70b-versatile`), with history per `conversation_id` | none locally, needs `GROQ_API_KEY` |
| TTS | `POST /api/v1/tts/synthesize` | [PocketTTS](https://huggingface.co/kyutai/pocket-tts) voice cloning from the clips in `TTS/assets/voice` | ~225 MB, fetched from Hugging Face on the first request |

Every module also has `GET .../health` and `GET .../config`. Models load lazily on the first request and stay in memory. If a model can't be loaded (package missing, no internet for the first download, gated Hugging Face repo, no API key), the route returns `503` with the cause. It never sends back a fake result.

## Local AI BMO inspiration

<a href="https://www.youtube.com/watch?v=l5ggH-YhuAw">
  <img src="https://i.ytimg.com/vi/l5ggH-YhuAw/maxresdefault.jpg" alt="Building a Real BMO — Embodied Local AI Agent video cover" width="520" />
</a>

This project shares the same local-AI spirit as [Building a Real BMO — Embodied Local AI Agent](https://www.youtube.com/watch?v=l5ggH-YhuAw): a Raspberry Pi-powered, voice-enabled BMO using local models. The cover above links directly to the build video.

## Setup

Verified on Windows 11 with Python 3.11 and [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv -r requirements-dev.txt
cp .env.example .env
```

Activate the venv with `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate`. `requirements.txt` pulls in the CPU build of PyTorch.

Two modules need accounts:

- **TTS**: voice cloning uses the gated `kyutai/pocket-tts` repo. Accept its terms at https://huggingface.co/kyutai/pocket-tts, then log in once with `hf auth login`.
- **LLM**: put a Groq key in `.env` as `GROQ_API_KEY` (create one at https://console.groq.com/keys). `GROQ_BASE_URL` can point at any OpenAI-compatible server instead.

### Assets

The voice-cloning clips ship with the repo in `TTS/assets/voice/` (`1.wav` to `8.wav`, 4.3 MB), so a fresh clone needs no extra fetch step. The other models download themselves on first use, as listed in the table above. Generated audio goes to `TTS/assets/out/`. Uploads and chat logs go to `STT/assets/`, `CV/assets/` and `LLM/assets/`. All of these are gitignored.

## Run

```bash
python main.py
```

The server listens on `http://127.0.0.1:8000`, with interactive docs at `/docs`. On Windows PowerShell, type `curl.exe` rather than `curl`.

### STT

```bash
curl -X POST http://127.0.0.1:8000/api/v1/stt/transcribe \
  -F "file=@TTS/assets/voice/1.wav" -F "language=auto"
```

```json
{"model": "base", "language": "en", "duration_seconds": 2.0,
 "transcription": "I don't say things I don't mean.",
 "segments": [{"start": 0.0, "end": 2.0, "text": "I don't say things I don't mean."}], "...": "..."}
```

`language` defaults to `STT_DEFAULT_LANGUAGE` (`en`). Pass `auto` to let Whisper detect it.

### CV

```bash
curl -X POST http://127.0.0.1:8000/api/v1/cv/analyze -F "file=@path/to/photo.jpg"
```

```json
{"model": "yolo26n", "image_size": [810, 1080],
 "detections": [{"label": "bus", "confidence": 0.8806, "bbox": [0, 230, 803, 751]},
                {"label": "person", "confidence": 0.872, "bbox": [49, 398, 240, 902]}],
 "summary": "Detected 4 person, 1 bus.", "...": "..."}
```

`bbox` is `[x1, y1, x2, y2]` in pixels. Detections below `CV_CONFIDENCE_THRESHOLD` (0.35) are dropped.

### LLM

```bash
curl -X POST http://127.0.0.1:8000/api/v1/llm/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Who are you?"}'
```

The response has `reply`, `conversation_id`, `turns` and token `usage`. To continue a conversation, send the `conversation_id` back with the next message. History is kept in memory and resets when the server restarts. Without a key, the route returns:

```json
{"detail": "LLM is not configured: set GROQ_API_KEY in .env (create a key at https://console.groq.com/keys)."}
```

### TTS

```bash
curl -X POST http://127.0.0.1:8000/api/v1/tts/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "BMO says hello", "mode": "small"}'
```

```json
{"model": "pocket-tts", "voice": "1.wav", "mode": "small",
 "file_url": "/api/v1/tts/files/tts_pocket_20261023T200101_3bb4114d.wav",
 "timings_seconds": {"engine_load_s": 17.2, "inference_s": 3.3, "total_s": 21.1}, "...": "..."}
```

Download the audio from `file_url`:

```bash
curl -o bmo.wav http://127.0.0.1:8000/api/v1/tts/files/tts_pocket_20261023T200101_3bb4114d.wav
```

`mode` picks a clip: `small` uses `1.wav`, `medium` uses `2.wav`, `large` uses `3.wav`. `voice` picks any clip in `TTS/assets/voice` by number, stem or file name, for example `"voice": "4"`. An unknown voice returns `422`.

## TTS command line

```bash
python speak.py "hello world"
python speak.py -m "use the medium clip"
python speak.py --voice 4 "use sample 4"
python speak.py -i
```

`-i` is interactive mode: it plays each line from memory without saving files. See [TTS/README.md](TTS/README.md) for all options.

## Tests

```bash
pytest
```

Each router has its own smoke tests (`tests/test_stt.py`, `test_cv.py`, `test_llm.py`, `test_tts.py`). They send real HTTP requests through FastAPI's `TestClient` and check the parsing, file handling, response shape and error codes. The model is the only thing replaced: a fake Whisper, YOLO or PocketTTS engine, or a mocked Groq HTTP transport. The tests need no downloads or API keys and finish in about a second.

## Other entry points

- `notebook_paths.setup_notebook()` sets up notebooks so reads and saves resolve from the project root:

  ```python
  from notebook_paths import setup_notebook

  paths = setup_notebook(chdir_to_root=True)
  ```

- `train.py`, `exercises.py` and `challenges.py` find and run scripts you add to the matching folders. None exist yet.
