"""Where the results table of a dataset is stored.

The heavy lifting is left to the dependencies: Polars reads and writes Parquet,
including its key-value metadata, and ``deltalake`` provides atomic, versioned,
concurrent appends on local disks and object stores. The two stores here only
add what those do not: carrying the :class:`~qpdk.models.datasets.metadata.DatasetMetadata`
in the files, and a readable error for an unresolved Git LFS pointer.

:class:`ParquetParts`
    A local directory of append-only ``*.parquet`` parts, each carrying the
    metadata in its footer. The default, used for curated datasets in Git LFS.
    One writer at a time; use a Delta table for concurrent writers.

:class:`DeltaStore`
    A `Delta Lake <https://delta.io>`_ table at a local path or an object-store
    URI such as ``gs://``, ``s3://``, or ``az://``. Each append is one atomic
    commit and a new table version, so a team can write sweep results straight to
    a cloud bucket and pin a model to an exact version. The metadata is kept on
    the ``value`` column of the table schema, since Delta tables accept no
    free-form table properties. Requires the ``delta`` extra.

Credentials are passed as ``storage_options``, in the keys Polars and
``deltalake`` accept (for example ``aws_endpoint_url``,
``google_service_account``, ``azure_storage_account_name``), or come from the
environment and application-default credentials of the cloud SDKs.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qpdk.models.datasets.metadata import METADATA_KEY, DatasetMetadata

if TYPE_CHECKING:
    import polars as pl

_LFS_POINTER_PREFIX = b"version https://git-lfs"


class LFSPointerError(FileNotFoundError):
    """A results file is an unresolved Git LFS pointer instead of Parquet data."""


def _check_not_lfs_pointer(path: Path) -> None:
    """Raise :class:`LFSPointerError` if ``path`` holds an unresolved LFS pointer."""
    with path.open("rb") as file:
        head = file.read(len(_LFS_POINTER_PREFIX))
    if head == _LFS_POINTER_PREFIX:
        msg = (
            f"{path} is a Git LFS pointer, not Parquet data. From the root of a source "
            'checkout run `git lfs install && git lfs pull --include "qpdk/models/datasets/data/**"`. '
            "Installed qpdk wheels already contain the data."
        )
        raise LFSPointerError(msg)


class ParquetParts:
    """Local, append-only directory of Parquet parts.

    Args:
        location: Directory of the parts.
    """

    def __init__(self, location: str | Path) -> None:
        """Point at a directory of parts; nothing is read yet."""
        self.location = Path(location)

    def __repr__(self) -> str:
        """Show the location."""
        return f"ParquetParts({str(self.location)!r})"

    @property
    def files(self) -> list[Path]:
        """Part files in a deterministic order."""
        return sorted(self.location.glob("*.parquet"))

    def metadata(self) -> DatasetMetadata | None:
        """Metadata stored in the parts, or ``None`` if there are none yet.

        Returns:
            The metadata shared by every part.

        Raises:
            ValueError: if the parts disagree or a part has no metadata.
        """
        import polars as pl  # ruff: ignore[import-outside-top-level]

        texts = set()
        for file in self.files:
            _check_not_lfs_pointer(file)
            if (text := pl.read_parquet_metadata(file).get(METADATA_KEY)) is None:
                msg = f"{file} has no {METADATA_KEY!r} metadata."
                raise ValueError(msg)
            texts.add(text)
        if len(texts) > 1:
            msg = f"Parts in {self.location} carry different dataset metadata."
            raise ValueError(msg)
        return DatasetMetadata.from_json(texts.pop()) if texts else None

    def scan(self, schema: pl.Schema) -> pl.LazyFrame:
        """Scan all parts with ``schema``.

        Returns:
            Lazy frame over every part.

        Raises:
            FileNotFoundError: if there are no parts.
        """
        import polars as pl  # ruff: ignore[import-outside-top-level]

        if not (files := self.files):
            msg = f"No *.parquet parts in {self.location}."
            raise FileNotFoundError(msg)
        for file in files:
            _check_not_lfs_pointer(file)
        return pl.scan_parquet(files, schema=schema)

    def write(
        self, frame: pl.DataFrame, metadata: DatasetMetadata, part: str | None
    ) -> str:
        """Publish ``frame`` as a new part without ever replacing an existing one.

        Returns:
            Path of the written part.

        Raises:
            FileExistsError: if the part already exists.
            ValueError: if the part is not a filename stem.
        """
        if part is not None and (
            not part or Path(part).name != part or part in {".", ".."} or "\\" in part
        ):
            raise ValueError("Part must be a filename stem without path separators")
        self.location.mkdir(parents=True, exist_ok=True)
        target = self.location / f"{part or uuid.uuid4().hex}.parquet"
        if target.exists():
            msg = f"{target} already exists; result parts are append-only."
            raise FileExistsError(msg)
        staging = self.location / f".{target.name}.{uuid.uuid4().hex}.tmp"
        try:
            frame.write_parquet(
                staging, statistics=True, metadata={METADATA_KEY: metadata.to_json()}
            )
            # A hard link publishes atomically and, unlike a rename, never
            # replaces an existing part.
            os.link(staging, target)
        finally:
            staging.unlink(missing_ok=True)
        return str(target)


class DeltaStore:
    """Results stored as a Delta Lake table.

    Every append is one atomic commit and creates a new table version. Reading a
    pinned ``version`` reproduces exactly the rows a model was built from.

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

    def _table(self) -> Any:
        """Open the table.

        Returns:
            The ``deltalake.DeltaTable``, or ``None`` if there is none yet.
        """
        import deltalake  # ruff: ignore[import-outside-top-level]

        try:
            return deltalake.DeltaTable(
                self.uri, version=self.version, storage_options=self.storage_options
            )
        except deltalake.exceptions.TableNotFoundError:
            return None

    def metadata(self) -> DatasetMetadata | None:
        """Metadata stored in the table schema, or ``None`` if there is no table.

        Returns:
            The table's dataset metadata.

        Raises:
            ValueError: if the table has no dataset metadata.
        """
        if (table := self._table()) is None:
            return None
        for field in table.schema().fields:
            if field.name == "value" and METADATA_KEY in (field.metadata or {}):
                return DatasetMetadata.from_json(field.metadata[METADATA_KEY])
        msg = f"Delta table at {self.uri} has no {METADATA_KEY!r} metadata."
        raise ValueError(msg)

    @property
    def table_version(self) -> int:
        """Version of the table that :meth:`scan` reads."""
        if (table := self._table()) is None:
            msg = f"No Delta table at {self.uri}."
            raise FileNotFoundError(msg)
        return table.version()

    def scan(self, schema: pl.Schema) -> pl.LazyFrame:
        """Scan the table, cast to ``schema``.

        Returns:
            Lazy frame over the table.

        Raises:
            FileNotFoundError: if there is no table.
        """
        import polars as pl  # ruff: ignore[import-outside-top-level]

        if (table := self._table()) is None:
            msg = f"No Delta table at {self.uri}."
            raise FileNotFoundError(msg)
        return pl.scan_delta(table).select(schema.names()).cast(schema)  # pyrefly: ignore[bad-argument-type]

    def write(
        self, frame: pl.DataFrame, metadata: DatasetMetadata, part: str | None
    ) -> str:
        """Commit ``frame`` as one append, creating the table if needed.

        New tables are partitioned by quantity and variants, which are selected
        exactly and never interpolated, and carry ``metadata`` on the ``value``
        column. Initial creation is exclusive; retry an append if another
        writer creates the table first.

        Returns:
            The URI and the committed table version.

        Raises:
            ValueError: if the store is pinned to a version.
        """
        import deltalake  # ruff: ignore[import-outside-top-level]
        from arro3.core import Table  # ruff: ignore[import-outside-top-level]

        if self.version is not None:
            msg = f"{self!r} is pinned to a version and read-only."
            raise ValueError(msg)
        table = self._table()
        exists = table is not None
        data: Any = frame
        if not exists:
            table = Table.from_arrow(frame)
            index = table.schema.get_field_index("value")
            field = table.schema.field(index).with_metadata({
                METADATA_KEY: metadata.to_json()
            })
            data = Table.from_batches(
                table.to_batches(), schema=table.schema.set(index, field)
            )
        deltalake.write_deltalake(
            table if exists else self.uri,
            data,
            mode="append" if exists else "error",
            storage_options=self.storage_options,
            **(
                {}
                if exists
                else {
                    "name": metadata.name,
                    "description": metadata.description,
                    "partition_by": ["quantity", *metadata.variants],
                }
            ),
            commit_properties=deltalake.CommitProperties(
                custom_metadata={"qpdk.part": part or ""}
            ),
        )
        # The writer updates this snapshot to its own commit, not a later append.
        return f"{self.uri}@v{table.version() if exists else 0}"
