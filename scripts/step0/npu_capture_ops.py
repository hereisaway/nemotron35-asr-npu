"""Check which op families break NPU graph capture (bisect helper).

Usage: PER_STREAM_QUEUE=1 TASK_QUEUE_ENABLE=1 python npu_capture_ops.py
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("PER_STREAM_QUEUE", "1")
os.environ.setdefault("TASK_QUEUE_ENABLE", "1")
os.environ.pop("ASCEND_LAUNCH_BLOCKING", None)

import torch  # noqa: E402
import torch_npu  # noqa: E402
import torch.nn.functional as F  # noqa: E402

DEVICE = torch.device("npu")
torch.npu.set_device(DEVICE)
# The graph backend disables internal formats before capture; mirror that here.
torch.npu.config.allow_internal_format = False


def try_capture(name, build):
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(npu_graph=graph):
            build()
        torch.npu.synchronize()
        return {"op": name, "captured": True}
    except Exception as error:  # noqa: BLE001
        return {
            "op": name,
            "captured": False,
            "error_type": type(error).__name__,
            "error": str(error).splitlines()[0][:400],
        }


def main() -> None:
    torch.manual_seed(0)
    linear = torch.nn.Linear(256, 256).to(DEVICE)
    conv = torch.nn.Conv1d(256, 256, 3, padding=1).to(DEVICE)
    lstm = torch.nn.LSTM(256, 256, num_layers=2, batch_first=True).to(DEVICE)
    x = torch.randn(4, 32, 256, device=DEVICE)

    results = [
        try_capture("linear", lambda: linear(x)),
        try_capture("conv1d", lambda: conv(x.transpose(1, 2))),
        try_capture("lstm", lambda: lstm(x)),
        try_capture(
            "matmul_softmax",
            lambda: torch.softmax(x @ x.transpose(1, 2), dim=-1),
        ),
        try_capture(
            "one_hot",
            lambda: F.one_hot(
                torch.zeros(4, 32, dtype=torch.long, device=DEVICE), num_classes=32
            ).to(x.dtype),
        ),
        try_capture(
            "index_add",
            lambda: torch.zeros(8, 8, device=DEVICE).index_add_(
                0,
                torch.tensor([0, 1, 2, 3], device=DEVICE),
                torch.ones(4, 8, device=DEVICE),
            ),
        ),
        try_capture(
            "sdpa",
            lambda: F.scaled_dot_product_attention(x, x, x),
        ),
    ]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
