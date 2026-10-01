"""Tests for qpdk.datasets.store: results in Delta Lake tables and object stores."""

import shutil
import socket
import subprocess
import sys
import time
import tomllib
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from qpdk.datasets import (
    DATASETS_PATH,
    Dataset,
    DatasetError,
    DeltaStore,
    Manifest,
    ParquetParts,
)

deltalake = pytest.importorskip("deltalake")

NAME = "plate_capacitor_synthetic"
BUCKET = "qpdk-results"


@pytest.fixture(scope="module")
def source() -> Dataset:
    return Dataset(NAME)


@pytest.fixture
def manifest_dir(tmp_path: Path) -> Path:
    """A dataset directory holding only the manifest, as for a cloud-backed dataset."""
    target = tmp_path / NAME
    target.mkdir()
    shutil.copy(DATASETS_PATH / NAME / "manifest.toml", target)
    return target


def _with_storage(manifest_dir: Path, storage: str) -> None:
    """Append a ``[storage]`` section to the manifest in ``manifest_dir``."""
    path = manifest_dir / "manifest.toml"
    path.write_text(path.read_text() + "\n[storage]\n" + storage)


def _halves(source: Dataset) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split the source table into two batches of whole runs."""
    first = pl.col("gap") < 10.0
    return source.table.filter(first), source.table.filter(~first)


def test_delta_round_trip_matches_parquet_parts(
    source: Dataset, manifest_dir: Path, tmp_path: Path
) -> None:
    dataset = Dataset(manifest_dir, store=DeltaStore(tmp_path / "table"))
    first, second = _halves(source)
    assert dataset.append(first, part="batch-0").endswith("@v0")
    assert dataset.append(second, part="batch-1").endswith("@v1")
    np.testing.assert_array_equal(
        dataset.grid("maxwell_capacitance", cross_section="cpw").values,
        source.grid("maxwell_capacitance", cross_section="cpw").values,
    )
    history = deltalake.DeltaTable(tmp_path / "table").history()
    assert [h["qpdk.part"] for h in history] == ["batch-1", "batch-0"]


def test_delta_rejects_duplicates_across_commits(
    source: Dataset, manifest_dir: Path, tmp_path: Path
) -> None:
    dataset = Dataset(manifest_dir, store=DeltaStore(tmp_path / "table"))
    first, _ = _halves(source)
    dataset.append(first)
    with pytest.raises(DatasetError, match="duplicate"):
        dataset.append(first.head(4))
    assert deltalake.DeltaTable(tmp_path / "table").version() == 0


def test_delta_version_pin_reproduces_old_rows(
    source: Dataset, manifest_dir: Path, tmp_path: Path
) -> None:
    uri = tmp_path / "table"
    first, second = _halves(source)
    writer = Dataset(manifest_dir, store=DeltaStore(uri))
    writer.append(first)
    writer.append(second)
    pinned = Dataset(manifest_dir, store=DeltaStore(uri, version=0))
    assert pinned.table.height == first.height
    assert writer.table.height == source.table.height
    with pytest.raises(ValueError, match="pinned to a version and read-only"):
        pinned.append(second)


def test_delta_refuses_another_datasets_table(
    source: Dataset, manifest_dir: Path, tmp_path: Path
) -> None:
    uri = tmp_path / "table"
    Dataset(manifest_dir, store=DeltaStore(uri)).append(_halves(source)[0])
    path = manifest_dir / "manifest.toml"
    path.write_text(path.read_text().replace(f'name = "{NAME}"', 'name = "other"', 1))
    other = Dataset(manifest_dir, store=DeltaStore(uri))
    with pytest.raises(ValueError, match=f"holds dataset '{NAME}', not 'other'"):
        other.scan()


def test_missing_delta_table_is_explicit(manifest_dir: Path, tmp_path: Path) -> None:
    dataset = Dataset(manifest_dir, store=DeltaStore(tmp_path / "nowhere"))
    with pytest.raises(FileNotFoundError, match="No Delta table"):
        dataset.scan()


def test_manifest_storage_section_selects_delta(
    source: Dataset, manifest_dir: Path
) -> None:
    _with_storage(manifest_dir, 'format = "delta"\nuri = "results.delta"\n')
    dataset = Dataset(manifest_dir)
    assert isinstance(dataset.store, DeltaStore)
    assert dataset.store.uri == str(manifest_dir / "results.delta")
    dataset.append(source.table)
    assert Dataset(manifest_dir).table.height == source.table.height


def test_manifest_storage_version_requires_delta(manifest_dir: Path) -> None:
    data = tomllib.loads((manifest_dir / "manifest.toml").read_text())
    data["storage"] = {"format": "parquet", "uri": "s3://b/x", "version": 1}
    with pytest.raises(ValueError, match="Only a 'delta' storage can pin"):
        Manifest.model_validate(data)


# ---------------------------------------------------------------------------
# S3-compatible object store, emulated locally by moto
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def s3() -> Iterator[dict[str, str]]:
    """Run a local S3-compatible server with one empty bucket.

    The server runs in a subprocess: ``deltalake`` blocks without releasing the
    GIL, which deadlocks moto's in-process threaded server.

    Yields:
        Storage options for the server, in the keys Polars and ``deltalake`` accept.
    """
    boto3 = pytest.importorskip("boto3")
    pytest.importorskip("moto.server")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        host, port = probe.getsockname()
    server = subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]
        [sys.executable, "-m", "moto.server", "-H", host, "-p", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    options = {
        "aws_endpoint_url": f"http://{host}:{port}",
        "aws_access_key_id": "testing",
        "aws_secret_access_key": "testing",
        "aws_region": "us-east-1",
        "aws_allow_http": "true",
        "conditional_put": "etag",
    }
    try:
        for _ in range(100):
            with socket.socket() as client:
                if client.connect_ex((host, port)) == 0:
                    break
            time.sleep(0.1)
        else:
            pytest.skip("Local S3 server did not start.")
        with pytest.MonkeyPatch.context() as env:
            # Talk to the local server directly, never through an HTTP proxy.
            for key in ("NO_PROXY", "no_proxy"):
                env.setenv(key, "127.0.0.1,localhost")
            boto3.client(
                "s3",
                endpoint_url=options["aws_endpoint_url"],
                aws_access_key_id=options["aws_access_key_id"],
                aws_secret_access_key=options["aws_secret_access_key"],
                region_name=options["aws_region"],
            ).create_bucket(Bucket=BUCKET)
            yield options
    finally:
        server.terminate()
        server.wait()


def test_delta_on_s3_compatible_store(
    s3: dict[str, str], source: Dataset, manifest_dir: Path
) -> None:
    _with_storage(manifest_dir, f'format = "delta"\nuri = "s3://{BUCKET}/{NAME}"\n')
    dataset = Dataset(manifest_dir, storage_options=s3)
    first, second = _halves(source)
    dataset.append(first)
    dataset.append(second)
    reader = Dataset(manifest_dir, storage_options=s3)
    np.testing.assert_array_equal(
        reader.grid("maxwell_capacitance", cross_section="cpw").values,
        source.grid("maxwell_capacitance", cross_section="cpw").values,
    )
    assert isinstance(reader.store, DeltaStore)
    assert reader.store.table_version(reader.manifest) == 1


def test_parquet_parts_read_from_s3_but_never_append(
    s3: dict[str, str], source: Dataset, manifest_dir: Path
) -> None:
    uri = f"s3://{BUCKET}/parts-{NAME}"
    for i, batch in enumerate(_halves(source)):
        batch.write_parquet(f"{uri}/part-{i:04d}.parquet", storage_options=s3)
    dataset = Dataset(manifest_dir, store=ParquetParts(uri, storage_options=s3))
    assert dataset.table.height == source.table.height
    with pytest.raises(NotImplementedError, match="Use a DeltaStore"):
        dataset.append(source.table.head(0))
