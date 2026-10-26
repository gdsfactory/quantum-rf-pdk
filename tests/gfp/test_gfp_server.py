"""GDSFactory+ CLI smoke test for the QPDK project."""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
from pathlib import Path
from threading import Timer

import pytest
from conftest import GFP_REQUIRED_ENV

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _find_gfp_binary() -> str:
    """Return the configured GDSFactory+ binary or skip local runs."""
    if binary := os.environ.get("GFP_BIN") or shutil.which("gfp"):
        return str(Path(binary).resolve())
    if os.environ.get(GFP_REQUIRED_ENV) == "1":
        pytest.fail("gfp binary not found; run `just fetch-gfp`")
    pytest.skip("gfp binary not found; run `just fetch-gfp`")
    raise AssertionError("unreachable")


@pytest.mark.gfp
def test_gfp_indexes_qpdk() -> None:
    """Load the project settings and index its PDK factories."""
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [_find_gfp_binary(), "--cwd", str(PROJECT_ROOT), "stats"],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    assert result.returncode == 0, result.stderr[-2000:]
    assert "Files:" in result.stdout
    assert "\nqpdk " in result.stdout


@pytest.mark.gfp
def test_gfp_loads_layer_stack(tmp_path: Path) -> None:
    """The viewer must receive the full layer response over the server protocol."""
    request = json.dumps({"id": 1, "method": "listLayers", "params": {}}).encode()
    with (tmp_path / "server.log").open("wb") as log:
        server = subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]
            [
                _find_gfp_binary(),
                "serve",
                "--project-root",
                str(PROJECT_ROOT),
                "--host",
                "127.0.0.1",
                "--port",
                "0",
            ],
            cwd=PROJECT_ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
        )
        timeout = Timer(300, server.kill)
        timeout.start()
        try:
            assert server.stdin is not None
            assert server.stdout is not None
            server.stdin.write(
                f"Content-Length: {len(request)}\r\n\r\n".encode() + request
            )
            server.stdin.flush()
            while True:
                content_length = 0
                while (line := server.stdout.readline()) not in {b"\r\n", b"\n"}:
                    assert line, (tmp_path / "server.log").read_text()[-2000:]
                    if line.startswith(b"Content-Length:"):
                        content_length = int(line.split(b":", 1)[1])
                response = json.loads(server.stdout.read(content_length))
                if response.get("type") == "response" and response.get("id") == 1:
                    break
            assert "error" not in response, response
            result = response["result"]
            assert result["inferred_default_cross_section"] == "coplanar_waveguide"
            stack = json.loads(base64.b64decode(result["layerstack_b64"]))
            assert stack["layers"]["M1"]
            assert stack["layers"]["Substrate"]["background"] is True
        finally:
            timeout.cancel()
            server.terminate()
            server.wait(timeout=30)
