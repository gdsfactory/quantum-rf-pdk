"""Server-side integration tests for gdsfactoryplus v2 (the ``gfp`` server).

Covers the server-side integration surface of gdsfactoryplus 2.0.0 against
this PDK:

- Settings loading: the server loads ``[tool.gdsfactoryplus]`` from the
  project ``pyproject.toml`` and serves it over the ``getSettings`` RPC
  (the Python SDK ``gdsfactoryplus.settings.get_settings()`` only reads
  defaults, so it is intentionally not used here).
- Indexing: ``gfp stats``/``gfp index`` discovers qpdk cells and the
  ``.pic.yml`` sample factories; the server exposes Python factories via
  ``listFactories``.
- Nyanlib generation: server startup writes ``build/models.nyanlib``.

The tests need the ``gfp`` binary, which is not pip-installable. Discovery:
``GFP_BIN`` environment variable, then ``gfp`` on ``PATH``. When the binary
is missing (or non-functional) these tests skip — even under
``GFP_REQUIRED=1``. ``GFP_REQUIRED=1`` (as set in the gfp CI jobs) still
turns the "package absent" and "v2 API missing" skips from
``conftest.import_gfp_module`` into failures; only the binary-dependent
tests keep their skip.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import signal
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

#: qpdk repository root (parent of ``tests/``).
PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Sentinel the stdout reader thread enqueues at EOF.
_EOF = object()

# Server tests share one ``gfp serve`` instance and the repository's
# ``build/`` directory, so they must never be split across xdist workers;
# run the suite with ``--dist loadgroup`` (test-gfp does).
pytestmark = pytest.mark.xdist_group("gfp-server")

#: ``.pic.yml`` factories that indexing must discover from ``qpdk/samples/``.
PIC_YAML_FACTORIES = (
    "qubit_test_chip",
    "flipmon_test_chip",
    "resonator_test_chip_yaml",
)

#: Well-known qpdk cells that must appear in the generated nyanlib.
NYANLIB_SPOT_CHECKS = (
    "models:qpdk.cells.resonator.quarter_wave_resonator_coupled",
    "models:qpdk.cells.transmon.double_pad_transmon",
)


# ---------------------------------------------------------------------------
# Minimal stdio JSON-RPC harness (adapted from the gfp integration tests)
# ---------------------------------------------------------------------------


@dataclass
class StdioRpc:
    """Content-Length framed JSON-RPC 2.0 client over stdin/stdout.

    A daemon reader thread parses framed messages off the server's
    stdout into a queue, so every wait enforces its own timeout. Direct
    blocking ``readline`` calls could hang the harness forever when the
    server stops emitting a full frame.
    """

    proc: subprocess.Popen
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _id: int = field(default=0)
    _messages: queue.Queue = field(default_factory=queue.Queue)
    _reader: threading.Thread = field(init=False)

    def __post_init__(self) -> None:
        """Start the daemon reader thread pumping stdout into the queue."""
        self._reader = threading.Thread(
            target=self._pump_stdout, name="gfp-rpc-reader", daemon=True
        )
        self._reader.start()

    def call(
        self, method: str, params: dict | None = None, timeout: float = 60
    ) -> dict:
        """Send a JSON-RPC request and return the matching response object."""
        with self._lock:
            self._id += 1
            req_id = self._id
            msg: dict = {"jsonrpc": "2.0", "id": req_id, "method": method}
            if params is not None:
                msg["params"] = params
            body = json.dumps(msg).encode()
            frame = f"Content-Length: {len(body)}\r\n\r\n".encode() + body
            self._write(frame)
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    error = f"Timed out waiting for response to request {req_id}"
                    raise TimeoutError(error)
                try:
                    incoming = self._messages.get(timeout=remaining)
                except queue.Empty:
                    error = f"Timed out waiting for response to request {req_id}"
                    raise TimeoutError(error) from None
                if incoming is _EOF:
                    error = (
                        f"gfp server closed its stdout before responding "
                        f"to request {req_id}"
                    )
                    raise EOFError(error)
                # Calls are serialized by _lock, so anything else is a
                # stale response or a server notification; drop it.
                if incoming.get("id") == req_id:
                    return incoming

    def _write(self, data: bytes) -> None:
        stdin = self.proc.stdin
        assert stdin is not None  # spawned with stdin=PIPE
        stdin.write(data)
        stdin.flush()

    def _pump_stdout(self) -> None:
        """Read framed JSON-RPC messages from stdout until EOF (daemon thread)."""
        stdout = self.proc.stdout
        assert stdout is not None  # spawned with stdout=PIPE
        while True:
            content_length = 0
            saw_header = False
            while line := stdout.readline():
                text = line.decode(errors="replace").strip()
                if not text:
                    break  # blank line ends the header block
                if text.lower().startswith("content-length:"):
                    content_length = int(text.split(":", 1)[1].strip())
                    saw_header = True
            else:
                break  # EOF between frames
            if not saw_header:
                continue
            body = stdout.read(content_length)
            if len(body) < content_length:
                break  # EOF mid-frame
            try:
                self._messages.put(json.loads(body))
            except json.JSONDecodeError:
                continue  # skip malformed frames; the request will time out
        self._messages.put(_EOF)


@dataclass
class GfpServer:
    """A running ``gfp serve`` process against the qpdk project."""

    rpc: StdioRpc
    process: subprocess.Popen
    project_root: Path
    log_path: Path


def _find_gfp_binary() -> str:
    """Locate the ``gfp`` binary (GFP_BIN env var, else ``gfp`` on PATH).

    The binary is not pip-installable, so a missing (or non-functional)
    binary always skips — even under ``GFP_REQUIRED=1``; only the Python
    package and its v2 API are required by that policy (see conftest).

    Returns:
        Path to the ``gfp`` binary.

    Raises:
        AssertionError: Unreachable; ``pytest.skip`` always raises before this.
    """
    if env := os.environ.get("GFP_BIN"):
        return str(Path(env).resolve())
    if path := shutil.which("gfp"):
        return path
    pytest.skip("gfp binary not found (set GFP_BIN or put gfp on PATH)")
    raise AssertionError("unreachable")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def spawn_gfp_server(
    gfp_bin: str, project_root: Path, startup_timeout: float = 300
) -> Generator[GfpServer, None, None]:
    """Spawn ``gfp serve`` against a project root and wait until it is ready.

    Ready means the server's own startup signals have settled: ``indexing``
    false and ``nyanlib_generating`` false in ``getInfo`` — at that point the
    startup nyanlib cycle has published ``build/models.nyanlib``.

    Yields:
        GfpServer: the running server with a ready stdio RPC client.

    """
    log_path = project_root / "build" / "test-gfp-serve.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_file = Path(log_path).open(  # ruff: ignore[open-file-with-context-handler]
        "w", encoding="utf-8"
    )

    try:
        port = _free_port()
        proc = subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]
            [gfp_bin, "serve", "--port", str(port)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log_file,
            cwd=str(project_root),
        )
    except BaseException:
        log_file.close()
        raise

    rpc = StdioRpc(proc)
    deadline = time.monotonic() + startup_timeout
    while time.monotonic() < deadline:
        try:
            result = rpc.call("getInfo", {}, timeout=30).get("result", {})
            if (
                not result.get("indexing", True)
                and not result.get("nyanlib_generating", True)
                and result.get("index_error") is None
            ):
                break
        except (EOFError, TimeoutError, OSError, json.JSONDecodeError):
            # Not ready yet (server still booting or a frame not yet
            # parseable): retry until the deadline. The poll() check below
            # turns an early server exit into a proper failure.
            pass
        if proc.poll() is not None:
            log_file.close()
            pytest.fail(
                f"gfp serve exited with code {proc.returncode} before becoming ready. "
                f"See {log_path}"
            )
        time.sleep(2)
    else:
        proc.terminate()
        proc.wait(timeout=5)
        log_file.close()
        pytest.fail(
            f"gfp serve did not become ready within {startup_timeout}s. See {log_path}"
        )

    yield GfpServer(rpc=rpc, process=proc, project_root=project_root, log_path=log_path)

    if proc.stdin:
        proc.stdin.close()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    log_file.close()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def gfp_bin() -> str:
    """Path to a functional ``gfp`` binary, discovered or skipped."""
    binary = _find_gfp_binary()
    try:
        subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            [binary, "hello"], capture_output=True, timeout=10, check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pytest.skip(f"gfp binary at {binary} is not functional")
    return binary


@pytest.fixture(scope="session")
def gfp_server(gfp_bin: str) -> Generator[GfpServer, None, None]:
    """One ``gfp serve`` instance shared by all server tests in this module.

    Deletes ``build/models.nyanlib`` first so the startup nyanlib generation
    cycle actually runs (the server republishes it before reporting ready).

    Yields:
        GfpServer: the running server, ready for RPC calls.

    """
    nyanlib_path = PROJECT_ROOT / "build" / "models.nyanlib"
    if nyanlib_path.exists():
        nyanlib_path.unlink()

    yield from spawn_gfp_server(gfp_bin, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.gfp
def test_settings_snapshot_from_server(gfp_server: GfpServer) -> None:
    """The server merges [tool.gdsfactoryplus] from pyproject.toml into getSettings."""
    resp = gfp_server.rpc.call("getSettings", {}, timeout=30)
    assert "result" in resp, f"getSettings failed: {resp}"
    settings = resp["result"]

    assert settings["name"] == "qpdk"
    assert settings["pdk"]["name"] == "qpdk"
    assert settings["drc"]["pdk"] == "quantum_rf"
    assert settings["drc"]["timeout"] == 300
    assert settings["sim"]["x"]["name"] == "f"
    assert settings["sim"]["x"]["min"] == 1
    assert settings["sim"]["x"]["max"] == 15
    assert settings["sim"]["x"]["num"] == 4001
    assert settings["sim"]["x"]["unit"] == "GHz"
    assert settings["livewire"]["has_vias"] is False


@pytest.mark.gfp
def test_nyanlib_generated_at_startup(gfp_server: GfpServer) -> None:
    """Server startup publishes a non-empty models.nyanlib with qpdk cells."""
    nyanlib_path = gfp_server.project_root / "build" / "models.nyanlib"
    assert nyanlib_path.is_file(), (
        f"server reported startup complete but {nyanlib_path} is missing. "
        f"See {gfp_server.log_path}"
    )
    assert nyanlib_path.stat().st_size > 0, (
        f"server reported startup complete but {nyanlib_path} is empty. "
        f"See {gfp_server.log_path}"
    )

    nyanlib = json.loads(nyanlib_path.read_text())
    assert isinstance(nyanlib, dict)
    for key in NYANLIB_SPOT_CHECKS:
        assert key in nyanlib, (
            f"{key} missing from models.nyanlib; "
            f"qpdk entries present: {sorted(k for k in nyanlib if '.qpdk.' in k)[:10]}"
        )
        assert nyanlib[key].get("name"), f"nyanlib entry {key} has no name"


@pytest.mark.gfp
def test_server_indexes_qpdk_cells(gfp_server: GfpServer) -> None:
    """ListFactories exposes indexed qpdk Python factories."""
    resp = gfp_server.rpc.call("listFactories", {}, timeout=120)
    assert "result" in resp, f"listFactories failed: {resp}"
    qualified_names = [
        factory.get("qualified_name") for factory in resp["result"]["factories"]
    ]
    for expected in (
        "qpdk.cells.transmon.double_pad_transmon",
        "qpdk.cells.resonator.quarter_wave_resonator_coupled",
    ):
        assert qualified_names.count(expected) >= 1, (
            f"{expected} not indexed; sample of indexed names: "
            f"{[n for n in qualified_names if n and '.qpdk.' in n][:10]}"
        )


@pytest.mark.gfp
def test_gfp_stats_subprocess(gfp_bin: str) -> None:
    """``gfp stats`` succeeds, indexes qpdk and discovers the pic.yml factories."""
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [gfp_bin, "--cwd", str(PROJECT_ROOT), "stats"],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, (
        f"gfp stats failed (rc={result.returncode}).\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr[-2000:]}"
    )

    # Index summary lands on stdout; per-factory tracing logs on stderr.
    assert "Files:" in result.stdout
    assert "\nqpdk " in result.stdout, (
        f"qpdk missing from stats library table:\n{result.stdout}"
    )

    for factory_name in PIC_YAML_FACTORIES:
        assert f'"name":"{factory_name}"' in result.stderr, (
            f"pic.yml factory {factory_name!r} not discovered by gfp stats"
        )
