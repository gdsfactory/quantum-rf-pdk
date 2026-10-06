"""Tests for qpdk.datasets.store: results in Delta Lake tables and object stores."""

import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from qpdk.datasets import Dataset, DatasetError, DeltaStore
from qpdk.datasets.metadata import METADATA_KEY
from qpdk.datasets.synthetic import PLATE_CAPACITOR

deltalake = pytest.importorskip("deltalake")

NAME = PLATE_CAPACITOR.name
BUCKET = "qpdk-results"


@pytest.fixture(scope="module")
def source() -> Dataset:
    return Dataset(NAME)


def _halves(source: Dataset) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split the source table into two batches of whole runs."""
    first = pl.col("gap") < 10.0
    return source.table.filter(first), source.table.filter(~first)


def _grid(dataset: Dataset) -> np.ndarray:
    return dataset.grid("maxwell_capacitance", cross_section="cpw").values


class TestDeltaStore:
    """A Delta table holds the results and the metadata of a dataset."""

    @staticmethod
    def test_round_trip_matches_parquet_parts(source: Dataset, tmp_path: Path) -> None:
        uri = tmp_path / "table"
        dataset = Dataset(uri, PLATE_CAPACITOR, delta=True)
        first, second = _halves(source)
        assert dataset.append(first, part="batch-0").endswith("@v0")
        assert dataset.append(second, part="batch-1").endswith("@v1")
        np.testing.assert_array_equal(_grid(dataset), _grid(source))
        history = deltalake.DeltaTable(uri).history()
        assert [h["qpdk.part"] for h in history] == ["batch-1", "batch-0"]

    @staticmethod
    def test_metadata_is_stored_on_the_value_column(
        source: Dataset, tmp_path: Path
    ) -> None:
        uri = tmp_path / "table"
        Dataset(uri, PLATE_CAPACITOR, delta=True).append(source.table)
        (value,) = (
            f for f in deltalake.DeltaTable(uri).schema().fields if f.name == "value"
        )
        assert value.metadata[METADATA_KEY] == PLATE_CAPACITOR.to_json()
        reopened = Dataset(uri, delta=True)
        assert reopened.metadata == PLATE_CAPACITOR
        assert deltalake.DeltaTable(uri).metadata().partition_columns == [
            "cross_section"
        ]

    @staticmethod
    def test_rejects_duplicates_across_commits(source: Dataset, tmp_path: Path) -> None:
        dataset = Dataset(tmp_path / "table", PLATE_CAPACITOR, delta=True)
        first, _ = _halves(source)
        dataset.append(first)
        with pytest.raises(DatasetError, match="duplicate"):
            dataset.append(first.head(4))
        assert deltalake.DeltaTable(tmp_path / "table").version() == 0

    @staticmethod
    def test_version_pin_reproduces_old_rows(source: Dataset, tmp_path: Path) -> None:
        uri = tmp_path / "table"
        first, second = _halves(source)
        writer = Dataset(uri, PLATE_CAPACITOR, delta=True)
        writer.append(first)
        writer.append(second)
        pinned = Dataset(uri, delta=True, version=0)
        assert pinned.table.height == first.height
        assert writer.table.height == source.table.height
        assert isinstance(writer.store, DeltaStore)
        assert writer.store.table_version == 1
        with pytest.raises(ValueError, match="pinned to a version and read-only"):
            pinned.append(second)

    @staticmethod
    def test_refuses_another_datasets_metadata(source: Dataset, tmp_path: Path) -> None:
        uri = tmp_path / "table"
        Dataset(uri, PLATE_CAPACITOR, delta=True).append(_halves(source)[0])
        other = PLATE_CAPACITOR.model_copy(update={"name": "other"})
        with pytest.raises(DatasetError, match="differs from the metadata stored"):
            Dataset(uri, other, delta=True)

    @staticmethod
    def test_missing_table_is_explicit(tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="pass metadata"):
            Dataset(tmp_path / "nowhere", delta=True)
        dataset = Dataset(tmp_path / "nowhere", PLATE_CAPACITOR, delta=True)
        with pytest.raises(FileNotFoundError, match="No Delta table"):
            dataset.scan()


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


class TestObjectStore:
    """Delta tables on an S3-compatible bucket, as for Google Cloud Storage or Azure."""

    @staticmethod
    def test_delta_on_s3_compatible_store(s3: dict[str, str], source: Dataset) -> None:
        uri = f"s3://{BUCKET}/{NAME}"
        dataset = Dataset(uri, PLATE_CAPACITOR, delta=True, storage_options=s3)
        first, second = _halves(source)
        dataset.append(first)
        dataset.append(second)
        reader = Dataset(uri, delta=True, storage_options=s3)
        assert reader.metadata == PLATE_CAPACITOR
        np.testing.assert_array_equal(_grid(reader), _grid(source))
        assert isinstance(reader.store, DeltaStore)
        assert reader.store.table_version == 1
