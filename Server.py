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
model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8", cpu_threads=4)
print("Whisper ready.\n")

def transcribe(audio_bytes: bytes) -> str:
    """Raw float32 PCM bytes → transcribed text without Disk I/O."""
    signal = np.frombuffer(audio_bytes, dtype=np.float32)
    # beam_size=1 is significantly faster
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
                try:
                    text = await asyncio.to_thread(transcribe, message)
                    await websocket.send(json.dumps({"text": text}))
                    if text.strip():
                        print(f'  ✅ "{text}"')
                    else:
                        print("  ◌ (silence)")
                except Exception as e:
                    print(f"  ❌ Transcription error: {e}")
                    await websocket.send(json.dumps({"error": str(e)}))
                    
    except websockets.exceptions.ConnectionClosed:
        print(f"[-] Disconnected: {addr}")
    except Exception as e:
        print(f"[-] Unexpected error with {addr}: {e}")

async def main():
    print(f"Listening on ws://{HOST}:{PORT}\n")
    # Disable ping/pong to prevent 'keepalive ping timeout' during heavy CPU load
    async with websockets.serve(
        handle_client, 
        HOST, 
        PORT, 
        max_size=10 * 1024 * 1024,
        ping_interval=None
    ):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
