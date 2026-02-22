"""
server.py  —  runs on VPS (Optimized for Speed)
Receives raw float32 PCM audio via WebSocket,
transcribes with Whisper in-memory, sends text back.
"""

import asyncio
import json
import numpy as np
from faster_whisper import WhisperModel
import websockets

# ── config ───────────────────────────────────────────────────────────────────
HOST          = "0.0.0.0"
PORT          = 8765
SAMPLE_RATE   = 16000
# "tiny" is extremely fast, "base" is a good balance.
WHISPER_MODEL = "base"
# ─────────────────────────────────────────────────────────────────────────────

print(f"Loading faster-whisper '{WHISPER_MODEL}'...")
# Optimized for CPU: int8 + multiple threads
# Increase cpu_threads if your VPS has more cores
model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8", cpu_threads=4)
print("Whisper ready.\n")

def transcribe(audio_bytes: bytes) -> str:
    """Raw float32 PCM bytes → transcribed text without Disk I/O."""
    # Convert bytes directly to numpy array (No temp files!)
    signal = np.frombuffer(audio_bytes, dtype=np.float32)

    # beam_size=1 is significantly faster than the default 5.
    # vad_filter ignores silence chunks instantly.
    segments, _ = model.transcribe(
        signal, 
        language="en", 
        beam_size=1, 
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    
    return " ".join(seg.text.strip() for seg in segments)

async def handle_client(websocket):
    addr = websocket.remote_address
    print(f"[+] Connected: {addr}")
    try:
        async for message in websocket:
            if isinstance(message, bytes):
                # Run transcription in a thread to keep the socket alive and responsive
                text = await asyncio.to_thread(transcribe, message)
                
                # Send result back immediately
                await websocket.send(json.dumps({"text": text}))
                if text.strip():
                    print(f'  ✅ "{text}"')
                    
    except websockets.exceptions.ConnectionClosed:
        print(f"[-] Disconnected: {addr}")

async def main():
    print(f"Listening on ws://{HOST}:{PORT}\n")
    # max_size=10MB handles plenty of audio
    async with websockets.serve(handle_client, HOST, PORT, max_size=10 * 1024 * 1024):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
