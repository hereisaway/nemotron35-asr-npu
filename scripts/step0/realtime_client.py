"""Send a 16 kHz PCM16 WAV file to the native /v1/realtime endpoint."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import wave

import websockets


async def run(
    host: str, port: int, audio_path: str, prefill_wait_s: float, overall_timeout_s: float
) -> None:
    with wave.open(audio_path, "rb") as audio:
        assert (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) == (
            1,
            2,
            16000,
        )
        pcm = audio.readframes(audio.getnframes())

    url = f"ws://{host}:{port}/v1/realtime"
    finals: list[str] = []
    deltas: list[str] = []

    async with websockets.connect(url, max_size=None) as websocket:
        print(json.loads(await websocket.recv())["type"], flush=True)
        await websocket.send(
            json.dumps(
                {
                    "event_id": "configure",
                    "type": "session.update",
                    "session": {"output_modalities": ["text"]},
                }
            )
        )
        print(json.loads(await websocket.recv())["type"], flush=True)

        async def receive() -> None:
            while True:
                event = json.loads(await websocket.recv())
                event_type = event.get("type")
                if event_type == "error":
                    print(f"server error: {event}", flush=True)
                    return
                if event_type == "response.output_text.delta":
                    deltas.append(event.get("delta", ""))
                if event_type == "response.output_text.done":
                    finals.append(event.get("text", ""))
                if event_type == "sglang.input_audio.drained":
                    return

        reader = asyncio.create_task(receive())
        chunks = [pcm[offset : offset + 640] for offset in range(0, len(pcm), 640)]
        for sequence, chunk in enumerate(chunks[:50]):
            await websocket.send(
                json.dumps(
                    {
                        "event_id": f"audio-{sequence}",
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode(),
                        "sglang": {"seq": sequence},
                    }
                )
            )
        # note (hereisaway): the first NPU window compiles kernels before it can consume the rest of the stream.
        await asyncio.sleep(prefill_wait_s)
        for sequence, chunk in enumerate(chunks[50:], start=50):
            await websocket.send(
                json.dumps(
                    {
                        "event_id": f"audio-{sequence}",
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode(),
                        "sglang": {"seq": sequence},
                    }
                )
            )
            await asyncio.sleep(0.02)
        await websocket.send(
            json.dumps({"event_id": "end", "type": "sglang.input_audio.end"})
        )
        await asyncio.wait_for(reader, timeout=overall_timeout_s)
        print(f"final={finals[-1] if finals else ''.join(deltas)!r}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--audio", default="tests/data/query_to_draw.wav")
    parser.add_argument("--prefill-wait-s", type=float, default=25.0)
    parser.add_argument("--overall-timeout-s", type=float, default=180.0)
    args = parser.parse_args()
    asyncio.run(
        asyncio.wait_for(
            run(
                args.host,
                args.port,
                args.audio,
                args.prefill_wait_s,
                args.overall_timeout_s,
            ),
            timeout=args.overall_timeout_s + 30.0,
        )
    )


if __name__ == "__main__":
    main()
