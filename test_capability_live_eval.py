from __future__ import annotations

import os
import subprocess
import sys

import pytest


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_CAPABILITY_TESTS", "").strip().lower() not in {"1", "true", "yes", "on"},
    reason="Set RUN_LIVE_CAPABILITY_TESTS=1 to run live domain capability evaluation.",
)
def test_live_capability_pass_rate():
    cmd = [sys.executable, "scripts/evaluate_capabilities.py"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        pytest.fail(f"Capability evaluation failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")

