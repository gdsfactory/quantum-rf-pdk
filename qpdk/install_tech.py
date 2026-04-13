"""Symlink tech to klayout."""

import shutil
import sys
from pathlib import Path

from qpdk.logger import configure_logger, logger


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
        logger.warning(f"Technology already exists at {dest}")
        return
    if dest.exists() or dest.is_symlink():
        logger.info(f"Removing existing technology files: {dest}")
        remove_path_or_dir(dest)
    try:
        dest.symlink_to(src, target_is_directory=True)
        link_type = "Symlinked"
    except OSError:
        shutil.copytree(src, dest)
        link_type = "Copied"
    logger.info(f"{link_type} technology files:")
    logger.info(f"  From: {src}")
    logger.info(f"  To:   {dest}")


if __name__ == "__main__":
    configure_logger(log_format="<level>{message}</level>")
    logger.info("Installing KLayout technology...")
    klayout_folder = "KLayout" if sys.platform == "win32" else ".klayout"
    home = Path.home()
    dest_folder = home / klayout_folder / "tech"
    dest_folder.mkdir(exist_ok=True, parents=True)
    cwd = Path(__file__).resolve().parent
    repo_root = cwd.parent
    src = repo_root / "qpdk" / "klayout"
    dest = dest_folder / "qpdk"
    make_link(src=src, dest=dest)
    logger.info("Successfully installed KLayout technology.")
