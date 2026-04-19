# TTS

Lightweight local text-to-speech to RVC voice conversion for BMO.

Pipeline:

`text -> Piper -> wav -> RVC -> wav`

using:

`conda activate tts-rvc`

This module supports:

- local CLI generation
- local FastAPI generation through `/api/v1/tts/synthesize`
- optional remote Groq TTS fallback

## Files

- `cli.py`: CLI entrypoint logic
- `core.py`: FastAPI router
- `local_pipeline.py`: shared local Piper -> RVC pipeline
- `weights/`: RVC model files
- `assets/piper/`: downloaded Piper voice files
- `assets/tmp/`: intermediate wav files
- `assets/out/`: final generated wav files

## Current model setup

RVC model files:

- `TTS/weights/BMO.pth`
- `TTS/weights/added_IVF548_Flat_nprobe_1_BMO_v2.index`

Default Piper voice:

- `en_US-lessac-medium`

## CLI

From the repo root:

```powershell
python speak.py "Hello world"
python speak.py -s "Fast mode"
python speak.py -m "Balanced mode"
python speak.py -l "Best quality mode"
python speak.py -i
python speak.py .\my_text.txt
```

Interactive mode (`-i`) now plays audio directly from memory and does not save an output file per line.

After activating your environment, use plain `python speak.py ...`.

## Modes

- `-s` / `small`: `pm`
- `-m` / `medium`: `harvest`
- `-l` / `large`: `rmvpe`

Current parameter mapping:

- `small`: `index_rate=0.5`, `filter_radius=3`, `protect=0.3`
- `medium`: `index_rate=0.7`, `filter_radius=5`, `protect=0.4`
- `large`: `index_rate=0.8`, `filter_radius=7`, `protect=0.5`

## Output

Final wav files are saved to:

`TTS/assets/out`

Examples:

- `TTS/assets/out/bmo_small.wav`
- `TTS/assets/out/bmo_medium.wav`
- `TTS/assets/out/bmo_large.wav`

## API

Route:

`POST /api/v1/tts/synthesize`

Example body:

```json
{
  "text": "BMO says hello",
  "mode": "large",
  "use_remote": false
}
```

Response includes:

- `file_path`
- `mode`
- `pipeline`
- `used_remote`
- `voice`
- `model`
- `timings_seconds`

## Environment

Relevant `.env` values:

```env
ASSETS_TTS_DIR=TTS/assets
TTS_LOCAL_PIPER_DOWNLOAD_DIR=TTS/assets/piper
TTS_LOCAL_TMP_DIR=TTS/assets/tmp
TTS_LOCAL_OUT_DIR=TTS/assets/out
TTS_LOCAL_USE_WARM_ENGINE=true
TTS_LOCAL_PRIME_ON_PREWARM=true
TTS_RVC_MODEL_PATH=TTS/weights/BMO.pth
TTS_RVC_INDEX_PATH=TTS/weights/added_IVF548_Flat_nprobe_1_BMO_v2.index
TTS_RVC_DEVICE=cpu
TTS_LOCAL_PIPER_MODEL=en_US-lessac-medium
```

## Notes

- In interactive mode, models are prewarmed to reduce per-line latency.
- Interactive mode uses a realtime profile (`small` + direct in-memory playback) for maximum speed.
- Keep `TTS_LOCAL_USE_WARM_ENGINE=true` for fastest repeated usage.
- On CPU, the first warmup can still be slow; a CUDA-capable GPU is the main way to get near real-time conversion.
- Piper voice files are downloaded automatically on first use.
- For long paragraphs, use a `.txt` file instead of inline shell text.
- If the converted voice sounds weak, the main limitation is usually the source TTS voice, not the RVC wiring.
- `large` usually gives the best result in this setup.
