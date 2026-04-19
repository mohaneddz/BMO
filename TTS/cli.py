from __future__ import annotations

import argparse
import contextlib
import io
import logging
from pathlib import Path
import sys
import warnings

if __package__ in {None, ""}:
    # Allow `python cli.py` from inside `TTS/` by exposing repo root imports.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from TTS.local_pipeline import (
    load_text_source,
    prewarm_local_tts_engine,
    run_local_tts_pipeline,
    run_local_tts_playback,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local BMO TTS -> RVC pipeline.")
    parser.add_argument("text", nargs="?", help="Inline text or a path to a UTF-8 text file.")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("-s", "--small", action="store_true", help="Fastest mode.")
    mode_group.add_argument("-m", "--medium", action="store_true", help="Balanced mode.")
    mode_group.add_argument("-l", "--large", action="store_true", help="Highest-quality mode.")
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Start interactive CLI mode (type text and press Enter to speak).",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Generate files only; do not play audio in local speakers.",
    )
    return parser


def _resolve_mode(args: argparse.Namespace) -> str | None:
    if args.small:
        return "small"
    if args.medium:
        return "medium"
    if args.large:
        return "large"
    return None


def _configure_quiet_logging() -> None:
    noisy_loggers = (
        "fairseq",
        "rvc_python",
        "numba",
        "parselmouth",
        "torch",
    )
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.ERROR)


@contextlib.contextmanager
def _suppress_library_output() -> object:
    sink = io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            yield


def _play_wav(path: Path) -> None:
    try:
        import winsound
    except ImportError as exc:
        raise RuntimeError("Audio playback is only available on Windows in this CLI.") from exc
    winsound.PlaySound(str(path), winsound.SND_FILENAME)


def _play_wav_bytes(wav_bytes: bytes) -> None:
    try:
        import winsound
    except ImportError as exc:
        raise RuntimeError("Audio playback is only available on Windows in this CLI.") from exc
    winsound.PlaySound(wav_bytes, winsound.SND_MEMORY)


def _print_timings(timings: dict[str, float] | None) -> None:
    if not timings:
        return
    piper_s = timings.get("piper_s", 0.0)
    rvc_s = timings.get("rvc_s", 0.0)
    total_s = timings.get("total_s", 0.0)
    print(f"Timing -> total={total_s:.2f}s piper={piper_s:.2f}s rvc={rvc_s:.2f}s")


def _run_once(text: str, mode: str | None, play_audio: bool, direct_playback: bool) -> None:
    if direct_playback:
        with _suppress_library_output():
            playback = run_local_tts_playback(text=text, mode=mode, realtime=True)
        print(f"Done -> direct playback (mode={playback.mode}, pipeline={playback.pipeline})")
        _print_timings(playback.timings_seconds)
        if play_audio:
            _play_wav_bytes(playback.wav_bytes)
        return

    with _suppress_library_output():
        result = run_local_tts_pipeline(text=text, mode=mode)
    print(f"Done -> {result.output_wav_path} (mode={result.mode})")
    _print_timings(result.timings_seconds)
    if play_audio:
        _play_wav(result.output_wav_path)


def _interactive_loop(mode: str | None, play_audio: bool) -> int:
    with _suppress_library_output():
        warm_seconds = prewarm_local_tts_engine()
    print(f"Models warmed in {warm_seconds:.2f}s.")
    print("Interactive mode started. Type text and press Enter.")
    print("Audio is played directly in memory (no output file is saved).")
    print("Commands: /exit, /quit")
    while True:
        try:
            raw = input("> ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting interactive mode.")
            return 0

        text = raw.strip()
        if not text:
            continue
        if text.lower() in {"/exit", "/quit"}:
            print("Exiting interactive mode.")
            return 0

        _run_once(text=text, mode=mode, play_audio=play_audio, direct_playback=True)


def main() -> int:
    _configure_quiet_logging()
    parser = build_parser()
    args = parser.parse_args()
    mode = _resolve_mode(args)
    play_audio = not args.no_play

    if args.interactive:
        return _interactive_loop(mode=mode, play_audio=play_audio)

    text = load_text_source(args.text)
    _run_once(text=text, mode=mode, play_audio=play_audio, direct_playback=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
