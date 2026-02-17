"""Symlink tech to klayout."""

import shutil
import sys
from pathlib import Path


def remove_path_or_dir(dest: Path) -> None:
    """Remove a path or directory."""
    if not dest.exists():
        raise FileNotFoundError(f"Path does not exist: {dest}")

    if dest.is_dir():
        shutil.rmtree(dest)
    else:
        dest.unlink()


def make_link(src: str | Path, dest: str | Path, overwrite: bool = True) -> None:
    """Make a symbolic link from src to dest."""
    src, dest = Path(src), Path(dest)
    if not src.exists():
        raise FileNotFoundError(f"{src} does not exist")

    if dest.exists() and not overwrite:
        print(f"{dest} already exists")
        return
    if dest.exists() or dest.is_symlink():
        print(f"removing {dest} already installed")
        remove_path_or_dir(dest)
    try:
        Path.symlink_to(src, dest, target_is_directory=True)
    except OSError:
        shutil.copytree(src, dest)
    print("link made:")
    print(f"From: {src}")
    print(f"To:   {dest}")


if __name__ == "__main__":
    klayout_folder = "KLayout" if sys.platform == "win32" else ".klayout"
    home = Path.home()
    dest_folder = home / klayout_folder / "tech"
    dest_folder.mkdir(exist_ok=True, parents=True)
    cwd = Path(__file__).resolve().parent
    repo_root = cwd.parent
    src = repo_root / "qpdk" / "klayout"
    dest = dest_folder / "qpdk"
    make_link(src=src, dest=dest)
