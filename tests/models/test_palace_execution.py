"""The exact Palace binary runs with retained output, including failures."""

import importlib
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest


@pytest.fixture
def helper(monkeypatch: pytest.MonkeyPatch):
    scripts = Path(__file__).resolve().parents[2] / "qpdk/models/datasets/recipes"
    monkeypatch.syspath_prepend(str(scripts))
    return importlib.import_module("_palace")


@pytest.mark.skip_windows
@pytest.mark.parametrize("status", [0, 2])
def test_direct_launch_retains_output(helper, tmp_path: Path, status: int) -> None:
    run = tmp_path / "case with spaces"
    run.mkdir()
    executable = tmp_path / "palace wrapper"
    executable.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$PWD" "$@"\n'
        f'printf "solver diagnostic\\n" >&2\nexit {status}\n'
    )
    executable.chmod(0o755)
    execution = {"use_apptainer": False, "palace_executable": executable}
    if status:
        with pytest.raises(subprocess.CalledProcessError) as error:
            helper._run_palace(execution, run, 3)
        assert error.value.returncode == status
    else:
        helper._run_palace(execution, run, 3)
    assert (run / "solver.log").read_text().splitlines() == [
        str(run),
        "-np",
        "3",
        "-nt",
        "1",
        "config.json",
        "solver diagnostic",
    ]
    assert not (run / "complete.json").exists()


def test_container_uses_selected_mpi_binary(
    helper, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PALACE_EXECUTABLE", "/unselected/palace")

    def launch(command, **kwargs):
        assert command[:2] == ["apptainer", "exec"]
        assert command[command.index("--env") + 1] == "OMP_NUM_THREADS=1"
        assert command[-2:] == ["palace-x86_64.bin", "config.json"]
        assert command[command.index("-np") + 1] == "16"
        assert kwargs["cwd"] == tmp_path
        assert kwargs["check"]
        kwargs["stdout"].write("native solver diagnostic\n")

    monkeypatch.setattr(helper.subprocess, "run", launch)
    helper._run_palace(
        {
            "use_apptainer": True,
            "palace_sif_path": tmp_path / "palace.sif",
            "container_binary": "palace-x86_64.bin",
        },
        tmp_path,
        16,
    )
    assert (tmp_path / "solver.log").read_text() == "native solver diagnostic\n"


@pytest.mark.parametrize("header", [b"\x7fELF", b"\xcf\xfa\xed\xfe"])
@pytest.mark.parametrize("name", ["palace", "palace-native.bin"])
@pytest.mark.parametrize("processes", [1, 7])
@pytest.mark.parametrize("bundled_mpi", [False, True])
def test_native_launch_uses_mpi_without_wrapper_flags(
    helper,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    header: bytes,
    name: str,
    processes: int,
    bundled_mpi: bool,
) -> None:
    executable = tmp_path / name
    executable.write_bytes(header)
    mpi = tmp_path / ("mpirun.exe" if sys.platform == "win32" else "mpirun")
    if bundled_mpi:
        mpi.write_text("#!/bin/sh\nexit 0\n")
        mpi.chmod(0o755)
    monkeypatch.setenv("OMP_NUM_THREADS", "19")
    monkeypatch.setenv("PALACE_EXECUTABLE", "/unselected/palace")

    def launch(command, **kwargs):
        if bundled_mpi:
            assert Path(command[0]).samefile(mpi)
        else:
            assert command[0] == "mpirun"
        assert command[1:] == ["-np", str(processes), str(executable), "config.json"]
        assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
        assert kwargs["env"]["PALACE_EXECUTABLE"] == "/unselected/palace"
        assert kwargs["cwd"] == tmp_path
        assert kwargs["stderr"] == subprocess.STDOUT
        assert kwargs["check"]
        kwargs["stdout"].write("native solver diagnostic\n")

    monkeypatch.setattr(helper.subprocess, "run", launch)
    helper._run_palace(
        {"use_apptainer": False, "palace_executable": executable}, tmp_path, processes
    )
    assert (tmp_path / "solver.log").read_text() == "native solver diagnostic\n"


def test_native_launch_failure_retains_output(
    helper, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "palace"
    executable.write_bytes(b"\x7fELF")

    def launch(command, **kwargs):
        kwargs["stdout"].write("native failure diagnostic\n")
        raise subprocess.CalledProcessError(2, command)

    monkeypatch.setattr(helper.subprocess, "run", launch)
    with pytest.raises(subprocess.CalledProcessError) as error:
        helper._run_palace(
            {"use_apptainer": False, "palace_executable": executable}, tmp_path, 4
        )
    assert error.value.returncode == 2
    assert (tmp_path / "solver.log").read_text() == "native failure diagnostic\n"
    assert not (tmp_path / "complete.json").exists()


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("matched", [False, True])
@pytest.mark.parametrize(
    "prior_libraries", [None, os.pathsep.join(("/user/lib", "/other/lib"))]
)
def test_discovery_binds_only_matching_libraries(
    helper,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    native: bool,
    matched: bool,
    prior_libraries: str | None,
) -> None:
    prefix = tmp_path / "cached runtime"
    libraries = prefix / "lib"
    libraries.mkdir(parents=True)
    binary_dir = (prefix if matched else tmp_path / "user runtime") / "bin"
    binary_dir.mkdir(parents=True)
    executable = binary_dir / "palace"
    executable.write_bytes(b"\x7fELF" if native else b"#!/bin/sh\n")
    mpi = binary_dir / ("mpirun.exe" if sys.platform == "win32" else "mpirun")
    mpi.write_text("#!/bin/sh\nexit 0\n")
    mpi.chmod(0o755)
    sdk = ModuleType("gsim.palace.runtime")
    resolve_binary = Mock(return_value=executable)
    resolve_libraries = Mock(return_value=libraries)
    monkeypatch.setattr(sdk, "resolve_palace_binary", resolve_binary, raising=False)
    monkeypatch.setattr(
        sdk, "resolve_palace_library_dir", resolve_libraries, raising=False
    )
    monkeypatch.setitem(sys.modules, "gsim.palace.runtime", sdk)
    if prior_libraries is None:
        monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    else:
        monkeypatch.setenv("LD_LIBRARY_PATH", prior_libraries)
    monkeypatch.setenv("PATH", "/user/bin")
    execution, provenance = helper.runtime(None, None, "palace-native.bin")
    resolve_binary.assert_called_once_with()
    resolve_libraries.assert_called_once_with()
    assert execution.get("palace_library_dir") == (libraries if matched else None)
    if not matched:
        assert "palace_library_dir" not in execution
    assert "palace_library_dir" not in provenance

    def launch(command, **kwargs):
        environment = kwargs["env"]
        if matched:
            expected = str(libraries)
            if prior_libraries:
                expected += f"{os.pathsep}{prior_libraries}"
            assert environment["LD_LIBRARY_PATH"] == expected
            assert environment["PATH"] == f"{binary_dir}{os.pathsep}/user/bin"
        else:
            assert environment.get("LD_LIBRARY_PATH") == prior_libraries
            assert environment["PATH"] == "/user/bin"
        if native:
            assert Path(command[0]).samefile(mpi)
            assert command[1:] == ["-np", "5", str(executable), "config.json"]
            assert environment["OMP_NUM_THREADS"] == "1"
        else:
            assert command == [
                str(executable),
                "-np",
                "5",
                "-nt",
                "1",
                "config.json",
            ]

    monkeypatch.setattr(helper.subprocess, "run", launch)
    helper._run_palace(execution, tmp_path, 5)


@pytest.mark.parametrize("container", [False, True])
def test_explicit_runtime_does_not_resolve_cached_libraries(
    helper, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, container: bool
) -> None:
    selected = tmp_path / "explicit-runtime"
    selected.write_bytes(b"\x7fELF")
    sdk = ModuleType("gsim.palace.runtime")
    resolve_binary = Mock(side_effect=AssertionError("unexpected binary discovery"))
    resolve_libraries = Mock(side_effect=AssertionError("unexpected library discovery"))
    monkeypatch.setattr(sdk, "resolve_palace_binary", resolve_binary, raising=False)
    monkeypatch.setattr(
        sdk, "resolve_palace_library_dir", resolve_libraries, raising=False
    )
    monkeypatch.setitem(sys.modules, "gsim.palace.runtime", sdk)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/user/lib")
    execution, _ = helper.runtime(
        None if container else selected, selected if container else None, "native.bin"
    )
    assert "palace_library_dir" not in execution
    resolve_binary.assert_not_called()
    resolve_libraries.assert_not_called()

    def launch(command, **kwargs):
        if container:
            assert kwargs["env"] is None
            assert command[:3] == ["apptainer", "exec", "--cleanenv"]
        else:
            assert kwargs["env"]["LD_LIBRARY_PATH"] == "/user/lib"

    monkeypatch.setattr(helper.subprocess, "run", launch)
    helper._run_palace(execution, tmp_path, 2)
