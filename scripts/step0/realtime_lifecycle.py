"""Exercise native /v1/realtime session lifecycles on an NPU server.

Scenarios (issue #4310):
  reuse         A, B, A: a blank-but-valid session must not poison the slot
  interleaved   two concurrent sessions; the reference one must stay correct
  reconnect     drop a session mid-stream, then start a fresh one
  cancel        closing the socket mid-window must not stall or crash the server

Note: native transcription sessions do not implement response.cancel (the
server answers ``not_supported``), so cancellation is expressed by closing the
WebSocket. B is an audio file the vendor/HF path transcribes as empty (#4385);
it is used on purpose to prove that an empty session does not leak state.

Usage:
  python realtime_lifecycle.py --port 8011 --json-out logs/lifecycle.json
  python realtime_lifecycle.py --only cancel --prefill-wait-s 25
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
import urllib.request
import wave

import websockets

CHUNK_BYTES = 640  # 20 ms of 16 kHz mono PCM16


def read_pcm(path: str) -> bytes:
    with wave.open(path, "rb") as audio:
        assert (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) == (
            1,
            2,
            16000,
        ), f"{path} must be 16 kHz mono PCM16"
        return audio.readframes(audio.getnframes())


def health(host: str, port: int) -> dict:
    with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=10) as reply:
        return json.loads(reply.read())


async def wait_settled(host: str, port: int, timeout_s: float = 30.0) -> dict:
    """Wait until no request or completion is still in flight."""
    deadline = time.monotonic() + timeout_s
    while True:
        state = health(host, port)
        if (
            state.get("pending_completions", 0) == 0
            and not state.get("request_states")
        ):
            return state
        if time.monotonic() > deadline:
            return state
        await asyncio.sleep(0.5)


class Session:
    def __init__(self, url: str, timeout_s: float) -> None:
        self.url = url
        self.timeout_s = timeout_s
        self.events: list[dict] = []
        self.deltas: list[str] = []
        self.finals: list[str] = []
        self.errors: list[dict] = []
        self.drained = asyncio.Event()
        self.websocket = None
        self.reader: asyncio.Task | None = None

    async def __aenter__(self) -> "Session":
        self.websocket = await websockets.connect(self.url, max_size=None)
        event = json.loads(
            await asyncio.wait_for(self.websocket.recv(), max(30.0, self.timeout_s))
        )
        assert event["type"] == "session.created", event
        await self.websocket.send(
            json.dumps(
                {
                    "event_id": "configure",
                    "type": "session.update",
                    "session": {"output_modalities": ["text"]},
                }
            )
        )
        event = json.loads(
            await asyncio.wait_for(self.websocket.recv(), max(30.0, self.timeout_s))
        )
        assert event["type"] == "session.updated", event
        self.reader = asyncio.create_task(self._read())
        return self

    async def __aexit__(self, *exc) -> None:
        if self.reader is not None:
            self.reader.cancel()
            await asyncio.gather(self.reader, return_exceptions=True)
        if self.websocket is not None:
            await self.websocket.close()

    async def _read(self) -> None:
        try:
            while True:
                event = json.loads(await self.websocket.recv())
                self.events.append(event)
                kind = event.get("type")
                if kind == "response.output_text.delta":
                    self.deltas.append(event.get("delta", ""))
                elif kind == "response.output_text.done":
                    self.finals.append(event.get("text", ""))
                elif kind == "error":
                    self.errors.append(event)
                elif kind == "sglang.input_audio.drained":
                    self.drained.set()
                    return
        except websockets.ConnectionClosed:
            return

    async def send_audio(
        self,
        pcm: bytes,
        *,
        prefill_chunks: int = 50,
        prefill_wait_s: float = 0.0,
        pace_s: float = 0.02,
        start_seq: int = 0,
        stop_after_chunks: int | None = None,
    ) -> int:
        chunks = [pcm[i : i + CHUNK_BYTES] for i in range(0, len(pcm), CHUNK_BYTES)]
        if stop_after_chunks is not None:
            chunks = chunks[:stop_after_chunks]
        seq = start_seq
        for index, chunk in enumerate(chunks):
            if index == prefill_chunks and prefill_wait_s:
                await asyncio.sleep(prefill_wait_s)
            await self.websocket.send(
                json.dumps(
                    {
                        "event_id": f"audio-{seq}",
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode(),
                        "sglang": {"seq": seq},
                    }
                )
            )
            seq += 1
            if pace_s:
                await asyncio.sleep(pace_s)
        return seq

    async def end_audio(self) -> None:
        await self.websocket.send(
            json.dumps({"event_id": "end", "type": "sglang.input_audio.end"})
        )

    async def wait_drained(self) -> None:
        await asyncio.wait_for(self.drained.wait(), self.timeout_s)

    @property
    def text(self) -> str:
        if self.finals:
            return self.finals[-1]
        return "".join(self.deltas)


async def transcribe(
    url: str,
    pcm: bytes,
    timeout_s: float,
    *,
    prefill_wait_s: float = 0.0,
    pace_s: float = 0.02,
) -> dict:
    started = time.perf_counter()
    async with Session(url, timeout_s) as session:
        await session.send_audio(pcm, prefill_wait_s=prefill_wait_s, pace_s=pace_s)
        await session.end_audio()
        await session.wait_drained()
        return {
            "text": session.text,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "errors": session.errors,
        }


async def scenario_reuse(
    url: str,
    pcm_a: bytes,
    pcm_b: bytes,
    timeout_s: float,
    prefill: float,
) -> dict:
    first = await transcribe(url, pcm_a, timeout_s, prefill_wait_s=prefill)
    blank = await transcribe(url, pcm_b, timeout_s)
    third = await transcribe(url, pcm_a, timeout_s)
    checks = {
        "a_non_empty": bool(first["text"].strip()),
        "a_repeatable": first["text"].strip() == third["text"].strip(),
        "blank_session_isolated": third["text"].strip() == first["text"].strip(),
        "no_errors": not (first["errors"] or blank["errors"] or third["errors"]),
    }
    return {
        "checks": checks,
        "a": first,
        "blank": blank,
        "a_again": third,
        "reference": first["text"].strip(),
    }


async def scenario_interleaved(
    url: str,
    pcm_a: bytes,
    pcm_b: bytes,
    timeout_s: float,
    reference: str,
) -> dict:
    async def one(pcm: bytes) -> dict:
        started = time.perf_counter()
        async with Session(url, timeout_s) as session:
            await session.send_audio(pcm)
            await session.end_audio()
            await session.wait_drained()
            return {
                "text": session.text,
                "elapsed_s": round(time.perf_counter() - started, 3),
                "errors": session.errors,
            }

    # One real stream and one empty-but-valid stream, admitted concurrently.
    good, blank = await asyncio.gather(one(pcm_a), one(pcm_b))
    checks = {
        "good_non_empty": bool(good["text"].strip()),
        "matches_reference": (
            good["text"].strip() == reference if reference else bool(good["text"].strip())
        ),
        "no_errors": not (good["errors"] or blank["errors"]),
    }
    return {"checks": checks, "good": good, "blank": blank}


async def scenario_reconnect(url: str, pcm: bytes, timeout_s: float) -> dict:
    # Drop the socket mid-stream without an end-of-audio event.
    async with Session(url, timeout_s) as session:
        await session.send_audio(pcm, stop_after_chunks=40, pace_s=0.005)
        await asyncio.sleep(0.2)
    recovered = await transcribe(url, pcm, timeout_s)
    return {
        "checks": {
            "recovered_non_empty": bool(recovered["text"].strip()),
            "no_errors": not recovered["errors"],
        },
        "recovered": recovered,
    }


async def scenario_cancel(url: str, pcm: bytes, timeout_s: float, prefill_wait_s: float) -> dict:
    # Close while a window is in flight: for a cold server this is the first
    # window, i.e. cancellation during TBE compilation.
    async with Session(url, timeout_s) as session:
        await session.send_audio(
            pcm,
            prefill_chunks=10_000,
            pace_s=0.0,
            stop_after_chunks=60,
        )
        await asyncio.sleep(0.5)
    try:
        after = await transcribe(url, pcm, timeout_s, prefill_wait_s=prefill_wait_s)
        follow_up_error = ""
    except Exception as error:  # noqa: BLE001 - a failure here is the observation
        after = {"text": "", "elapsed_s": 0.0, "errors": []}
        follow_up_error = f"{type(error).__name__}: {error}"
    return {
        "checks": {
            "follow_up_non_empty": bool(after["text"].strip()),
            "no_errors": not follow_up_error and not after["errors"],
        },
        "follow_up": after,
        "follow_up_error": follow_up_error,
    }


SCENARIOS = ("reuse", "interleaved", "reconnect", "cancel")


async def run(args: argparse.Namespace) -> dict:
    url = f"ws://{args.host}:{args.port}/v1/realtime"
    pcm_a = read_pcm(args.audio_a)
    pcm_b = read_pcm(args.audio_b)
    only = set(args.only.split(",")) if args.only else set(SCENARIOS)
    unknown = only - set(SCENARIOS)
    assert not unknown, f"unknown scenarios: {sorted(unknown)}"

    report: dict = {
        "host": args.host,
        "port": args.port,
        "audio_a": args.audio_a,
        "audio_b": args.audio_b,
        "health_before": health(args.host, args.port),
        "scenarios": {},
    }
    reference = ""
    # The first realtime window compiles TBE kernels; give it room to finish.
    prefill = args.prefill_wait_s
    for name in SCENARIOS:
        if name not in only:
            continue
        started = time.perf_counter()
        if name == "reuse":
            result = await scenario_reuse(url, pcm_a, pcm_b, args.timeout_s, prefill)
            reference = result["reference"]
        elif name == "interleaved":
            result = await scenario_interleaved(
                url, pcm_a, pcm_b, args.timeout_s, reference
            )
        elif name == "reconnect":
            result = await scenario_reconnect(url, pcm_a, args.timeout_s)
        else:
            result = await scenario_cancel(url, pcm_a, args.timeout_s, prefill)
        result["wall_s"] = round(time.perf_counter() - started, 3)
        result["passed"] = all(result["checks"].values())
        report["scenarios"][name] = result
        report[f"health_after_{name}"] = await wait_settled(args.host, args.port)
        prefill = 0.0
        print(f"[{name}] passed={result['passed']} checks={result['checks']}", flush=True)
    report["health_after"] = await wait_settled(args.host, args.port)
    report["passed"] = all(s["passed"] for s in report["scenarios"].values())
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--audio-a", default="tests/data/query_to_draw.wav")
    parser.add_argument("--audio-b", default="/tmp/lifecycle_b.wav")
    parser.add_argument("--prefill-wait-s", type=float, default=25.0)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--only", default="")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    report = asyncio.run(run(args))
    payload = json.dumps(report, indent=2)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    print(payload)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
