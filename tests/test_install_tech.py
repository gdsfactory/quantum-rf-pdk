"""Tests for qpdk.install_tech module."""

from pathlib import Path

import pytest

from qpdk.install_tech import make_link, remove_path_or_dir


class TestRemovePathOrDir:
    """Tests for remove_path_or_dir."""

    @staticmethod
    def test_remove_file(tmp_path: Path) -> None:
        """Test removing a regular file."""
        f = tmp_path / "test.txt"
        f.write_text("hello")
        assert f.exists()
        remove_path_or_dir(f)
        assert not f.exists()

    @staticmethod
    def test_remove_directory(tmp_path: Path) -> None:
        """Test removing a directory tree."""
        d = tmp_path / "subdir"
        d.mkdir()
        (d / "file.txt").write_text("data")
        assert d.exists()
        remove_path_or_dir(d)
        assert not d.exists()

    @staticmethod
    def test_nonexistent_raises(tmp_path: Path) -> None:
        """Test that removing a nonexistent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="does not exist"):
            remove_path_or_dir(tmp_path / "nonexistent")


class TestMakeLink:
    """Tests for make_link."""

    @staticmethod
    def test_creates_symlink(tmp_path: Path) -> None:
        """Test that make_link creates a working symlink."""
        src = tmp_path / "source"
        src.mkdir()
        (src / "file.txt").write_text("data")
        dest = tmp_path / "link"

        make_link(src, dest)
        assert dest.exists()
        assert (dest / "file.txt").read_text() == "data"

    @staticmethod
    def test_overwrite_existing(tmp_path: Path) -> None:
        """Test that make_link overwrites an existing destination."""
        src = tmp_path / "source"
        src.mkdir()
        (src / "new.txt").write_text("new")

        dest = tmp_path / "link"
        dest.mkdir()
        (dest / "old.txt").write_text("old")

        make_link(src, dest)
        assert (dest / "new.txt").exists()

    @staticmethod
    def test_no_overwrite_skips(tmp_path: Path) -> None:
        """Test that overwrite=False skips existing destinations."""
        src = tmp_path / "source"
        src.mkdir()
        dest = tmp_path / "link"
        dest.mkdir()
        (dest / "old.txt").write_text("old")

        make_link(src, dest, overwrite=False)
        # Original dir should still be there unchanged
        assert (dest / "old.txt").exists()

    @staticmethod
    def test_source_not_found_raises(tmp_path: Path) -> None:
        """Test that a nonexistent source raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="does not exist"):
            make_link(tmp_path / "nonexistent", tmp_path / "link")
