"""Refuse an already-initialized CUDA runtime. Does not import torch to check."""

from __future__ import annotations

import os
import sys


def assert_cpu_only() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch = sys.modules.get("torch")
    if torch is None:
        return
    cuda = getattr(torch, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    if callable(is_available) and is_available():
        raise RuntimeError(
            "cpu-llm refuses to run: torch is imported and torch.cuda.is_available() is true"
        )
