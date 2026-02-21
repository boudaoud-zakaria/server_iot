"""
server.py  —  runs on VPS
Receives raw float32 PCM audio via WebSocket,
transcribes with Whisper, sends text back to client.
"""

import asyncio
import json
import tempfile
import os
import numpy as np
import soundfile as sf
import whisper
import websockets

# ── config ───────────────────────────────────────────────────────────────────
HOST          = "0.0.0.0"
PORT          = 8765
SAMPLE_RATE   = 16000
# Whisper model: "tiny" | "base" | "small" | "medium" | "large"
# "base" is the best balance for a typical VPS
WHISPER_MODEL = "base"
# ─────────────────────────────────────────────────────────────────────────────

print(f"Loading Whisper '{WHISPER_MODEL}'... (downloads on first run)")
model = whisper.load_model(WHISPER_MODEL)
print("Whisper ready.\n")


def transcribe(audio_bytes: bytes) -> str:
    """Raw float32 PCM bytes → transcribed text."""
    signal = np.frombuffer(audio_bytes, dtype=np.float32)

    # Whisper works best with a real file on disk
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    try:
        sf.write(tmp_path, signal, SAMPLE_RATE)
        result = model.transcribe(tmp_path, language="en", fp16=False)
        return result["text"].strip()
    finally:
        os.unlink(tmp_path)


async def handle_client(websocket):
    addr = websocket.remote_address
    print(f"[+] Connected: {addr}")
    try:
        async for message in websocket:
            if isinstance(message, bytes):
                print(f"  ⏳ Transcribing {len(message) // 4} samples...")
                # run in thread — keeps event loop unblocked
                text = await asyncio.to_thread(transcribe, message)
                print(f'  ✅ "{text}"')
                await websocket.send(json.dumps({"text": text}))
    except websockets.exceptions.ConnectionClosed:
        print(f"[-] Disconnected: {addr}")


async def main():
    print(f"Listening on ws://{HOST}:{PORT}\n")
    # max_size=10MB handles up to ~30s of audio
    async with websockets.serve(handle_client, HOST, PORT, max_size=10 * 1024 * 1024):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
