"""Where the results table of a dataset is stored.

The manifest always stays a small ``manifest.toml`` in Git; only the results
table moves. Two stores are provided:

:class:`ParquetParts`
    A directory of append-only ``*.parquet`` parts. The default, used for curated
    datasets in Git LFS. Reads also work from any Polars-compatible URI such as
    ``s3://``, ``gs://``, ``az://``, or ``https://``; appends are local only.

:class:`DeltaStore`
    A `Delta Lake <https://delta.io>`_ table at a local path or object-store URI.
    Each append is an atomic, versioned commit, so a team can write sweep results
    straight to a cloud bucket and pin a model to an exact table version.
    Requires the ``delta`` extra.

Credentials never go in the manifest. Pass them as ``storage_options``, in the
same keys Polars and ``deltalake`` accept (for example ``aws_endpoint_url``,
``google_service_account``, ``azure_storage_account_name``), or rely on the
environment and application-default credentials of the cloud SDKs.
"""

import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Protocol

import polars as pl

from qpdk.datasets.manifest import SCHEMA_VERSION, Manifest

_LFS_POINTER_PREFIX = b"version https://git-lfs"


class LFSPointerError(FileNotFoundError):
    """A results file is an unresolved Git LFS pointer instead of Parquet data."""


class ResultStore(Protocol):
    """Storage backend for the results table of one dataset."""

    def scan(self, schema: pl.Schema, manifest: Manifest) -> pl.LazyFrame:
        """Lazily read every stored row with ``schema``."""
        ...

    def lock(self) -> Any:
        """Context manager held while an append validates and writes."""
        ...

    def write(self, frame: pl.DataFrame, manifest: Manifest, part: str | None) -> str:
        """Durably add ``frame``; return a description of what was written."""
        ...


def is_uri(location: str | Path) -> bool:
    """Whether ``location`` is a URI such as ``s3://bucket/key`` rather than a path.

    Returns:
        True for ``scheme://`` locations.
    """
    return "://" in str(location)


def _check_not_lfs_pointer(path: Path) -> None:
    """Raise :class:`LFSPointerError` if ``path`` holds an unresolved LFS pointer."""
    with path.open("rb") as file:
        head = file.read(len(_LFS_POINTER_PREFIX))
    if head == _LFS_POINTER_PREFIX:
        msg = (
            f"{path} is a Git LFS pointer, not Parquet data. From the root of a source "
            'checkout run `git lfs install && git lfs pull --include "qpdk/datasets/data/**"`. '
            "Installed qpdk wheels already contain the data."
        )
        raise LFSPointerError(msg)


class ParquetParts:
    """Append-only directory of Parquet parts.

    Args:
        location: Local directory or Polars-compatible URI of the parts.
        storage_options: Object-store options for a URI, passed to Polars.
    """

    def __init__(
        self, location: str | Path, *, storage_options: dict[str, str] | None = None
    ) -> None:
        """Point at a directory of parts; nothing is read yet."""
        self.location = location if is_uri(location) else Path(location)
        self.storage_options = storage_options

    def __repr__(self) -> str:
        """Show the location."""
        return f"ParquetParts({str(self.location)!r})"

    @property
    def files(self) -> list[Path]:
        """Local part files in a deterministic order; empty for a URI."""
        if isinstance(self.location, Path):
            return sorted(self.location.glob("*.parquet"))
        return []

    def scan(self, schema: pl.Schema, manifest: Manifest) -> pl.LazyFrame:
        """Scan all parts with ``schema``.

        Returns:
            Lazy frame over every part.

        Raises:
            FileNotFoundError: if a local directory has no parts.
        """
        if not isinstance(self.location, Path):
            return pl.scan_parquet(
                f"{str(self.location).rstrip('/')}/*.parquet",
                schema=schema,
                storage_options=self.storage_options,
            )
        if not (files := self.files):
            msg = f"No results/*.parquet files for dataset {manifest.name!r} in {self.location}."
            raise FileNotFoundError(msg)
        for file in files:
            _check_not_lfs_pointer(file)
        return pl.scan_parquet(files, schema=schema)

    def lock(self) -> Any:
        """Hold ``.append.lock`` in the parts directory.

        Returns:
            Context manager for the lock.
        """
        location = self._local("append to")
        location.mkdir(parents=True, exist_ok=True)
        return _lock_file(location / ".append.lock")

    def write(
        self,
        frame: pl.DataFrame,
        manifest: Manifest,  # ruff: ignore[unused-method-argument]
        part: str | None,
    ) -> str:
        """Publish ``frame`` as a new part without ever replacing an existing one.

        Returns:
            Path of the written part.

        Raises:
            FileExistsError: if the part already exists.
        """
        location = self._local("append to")
        target = location / f"{part or uuid.uuid4().hex}.parquet"
        if target.exists():
            msg = f"{target} already exists; result parts are append-only."
            raise FileExistsError(msg)
        staging = location / f".{target.name}.{uuid.uuid4().hex}.tmp"
        try:
            frame.write_parquet(staging, compression="zstd", statistics=True)
            # A hard link publishes atomically and, unlike a rename, never
            # replaces an existing part.
            os.link(staging, target)
        finally:
            staging.unlink(missing_ok=True)
        return str(target)

    def _local(self, action: str) -> Path:
        """The parts directory, if it is local.

        Returns:
            Local directory.

        Raises:
            NotImplementedError: for a URI; use :class:`DeltaStore` to write remotely.
        """
        if not isinstance(self.location, Path):
            msg = (
                f"Cannot {action} Parquet parts at {self.location}: plain object stores "
                "give no atomic, non-overwriting publish. Use a DeltaStore to "
                "write results to a bucket."
            )
            raise NotImplementedError(msg)
        return self.location


@contextmanager
def _lock_file(path: Path) -> Iterator[None]:
    """Hold an exclusive lock file for the duration of an append.

    Yields:
        Nothing; the lock is released on exit.

    Raises:
        FileExistsError: if another append holds the lock.
    """
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        msg = (
            f"Another append holds {path}; retry, or delete it if no append is running."
        )
        raise FileExistsError(msg) from error
    try:
        os.close(fd)
        yield
    finally:
        path.unlink(missing_ok=True)


class DeltaStore:
    """Results stored as a Delta Lake table.

    Every append is one atomic commit and creates a new table version. Reading a
    pinned ``version`` reproduces exactly the rows a model was built from.

    The table records the dataset name, and a store refuses to read or write a
    table created for a different dataset.

    Concurrent appends are committed atomically on local disks, Google Cloud
    Storage, and Azure. On S3, pass ``{"conditional_put": "etag"}`` (S3 and most
    S3-compatible stores support it) or configure a ``deltalake`` locking
    provider. Validation runs before each commit and again on every read, so a
    duplicate committed by a racing writer is reported rather than silently used.

    Args:
        uri: Local path or object-store URI of the table, e.g. ``gs://bucket/dataset``.
        storage_options: Object-store options, passed to Polars and ``deltalake``.
        version: Table version to read; ``None`` reads the latest. A pinned store
            is read-only.
    """

    def __init__(
        self,
        uri: str | Path,
        *,
        storage_options: dict[str, str] | None = None,
        version: int | None = None,
    ) -> None:
        """Point at a table; nothing is read yet."""
        self.uri = str(uri)
        self.storage_options = storage_options
        self.version = version

    def __repr__(self) -> str:
        """Show the URI and pinned version."""
        pinned = "" if self.version is None else f", version={self.version}"
        return f"DeltaStore({self.uri!r}{pinned})"

    def _open(self, manifest: Manifest) -> Any:
        """Open the table and check that it belongs to ``manifest``.

        Returns:
            The ``deltalake.DeltaTable``.

        Raises:
            FileNotFoundError: if there is no table at the URI.
            ValueError: if the table belongs to another dataset.
        """
        import deltalake  # ruff: ignore[import-outside-top-level]

        try:
            table = deltalake.DeltaTable(
                self.uri, version=self.version, storage_options=self.storage_options
            )
        except deltalake.exceptions.TableNotFoundError as error:
            msg = f"No Delta table for dataset {manifest.name!r} at {self.uri}."
            raise FileNotFoundError(msg) from error
        if (name := table.metadata().name) != manifest.name:
            msg = f"Delta table at {self.uri} holds dataset {name!r}, not {manifest.name!r}."
            raise ValueError(msg)
        return table

    def table_version(self, manifest: Manifest) -> int:
        """Version of the table that :meth:`scan` reads.

        Returns:
            Delta table version.
        """
        return self._open(manifest).version()

    def scan(self, schema: pl.Schema, manifest: Manifest) -> pl.LazyFrame:
        """Scan the table, cast to ``schema``.

        Returns:
            Lazy frame over the table.
        """
        frame = pl.scan_delta(self._open(manifest))
        return frame.select(schema.names()).cast(schema)  # pyrefly: ignore[bad-argument-type]

    def lock(self) -> Any:
        """No client-side lock: commits are atomic in the Delta log.

        Returns:
            A no-op context manager.

        Raises:
            ValueError: if the store is pinned to a version.
        """
        if self.version is not None:
            msg = f"{self!r} is pinned to a version and read-only."
            raise ValueError(msg)
        return nullcontext()

    def write(self, frame: pl.DataFrame, manifest: Manifest, part: str | None) -> str:
        """Commit ``frame`` as one append, creating the table if needed.

        New tables are partitioned by the manifest variants, which are always
        selected exactly and never interpolated.

        Returns:
            The URI and the committed table version.
        """
        import deltalake  # ruff: ignore[import-outside-top-level]

        exists = deltalake.DeltaTable.is_deltatable(self.uri, self.storage_options)
        if exists:
            self._open(manifest)
        deltalake.write_deltalake(
            self.uri,
            frame,
            mode="append",
            storage_options=self.storage_options,
            **(
                {}
                if exists
                else {
                    "name": manifest.name,
                    "description": manifest.description,
                    "partition_by": [v.name for v in manifest.variants] or None,
                }
            ),
            commit_properties=deltalake.CommitProperties(
                custom_metadata={
                    "qpdk.schema_version": str(SCHEMA_VERSION),
                    "qpdk.part": part or "",
                }
            ),
        )
        return f"{self.uri}@v{self.table_version(manifest)}"
