# BMO

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

## Local TTS -> RVC Pipeline

This project now supports a CPU-only local speech pipeline for BMO:

`text -> Piper -> wav -> RVC -> wav`

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
```

### Modes

- `small`: `pm`, `index_rate=0.5`, `filter_radius=3`, `protect=0.3`
- `medium`: `harvest`, `index_rate=0.7`, `filter_radius=5`, `protect=0.4`
- `large`: `rmvpe`, `index_rate=0.8`, `filter_radius=7`, `protect=0.5`

### Default BMO RVC model paths

- `TTS/weights/BMO.pth`
- `TTS/weights/added_IVF548_Flat_nprobe_1_BMO_v2.index`

## Setup

### WSL development

Use WSL for local development on this Windows machine.

```bash
sudo apt update
sudo apt install -y ffmpeg espeak-ng

conda create -n tts-rvc python=3.10 -y
conda activate tts-rvc

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Raspberry Pi deployment

The deployment target is a headless Raspberry Pi running the FastAPI service locally.

```bash
sudo apt update
sudo apt install -y ffmpeg espeak-ng

conda create -n tts-rvc python=3.10 -y
conda activate tts-rvc

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Relevant environment variables

Configure these in `.env`:

```env
TTS_PROVIDER=local
TTS_USE_REMOTE=false
TTS_LOCAL_ENABLED=true
TTS_LOCAL_DEFAULT_MODE=small
TTS_LOCAL_PIPER_MODEL=en_US-lessac-medium
TTS_LOCAL_PIPER_DOWNLOAD_DIR=TTS/assets/piper
TTS_LOCAL_TMP_DIR=TTS/assets/tmp
TTS_LOCAL_OUT_DIR=TTS/assets/out
TTS_LOCAL_KEEP_TMP=false
TTS_RVC_MODEL_PATH=TTS/weights/BMO.pth
TTS_RVC_INDEX_PATH=TTS/weights/added_IVF548_Flat_nprobe_1_BMO_v2.index
TTS_RVC_DEVICE=cpu
```

## API example

```json
{
  "text": "BMO says hello",
  "mode": "medium",
  "use_remote": false
}
```
