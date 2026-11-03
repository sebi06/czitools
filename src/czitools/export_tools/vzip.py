"""Create reference-backed vzip archives without copying CZI pixels."""

import os
from pathlib import Path
import tempfile

from czitools.utils import logging_tools

logger = logging_tools.set_logging()


def create_czi_vzip(
    filepath: str | os.PathLike[str],
    output_path: str | os.PathLike[str] | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    """Save a CZI's virtual OME-Zarr hierarchy as a vzip archive.

    The archive contains metadata and references to the original CZI, not a
    copy of its pixels. The CZI must remain unchanged at its original location.
    This synchronous, GUI-independent function can be called from a worker.
    Output is published only after refkv finishes writing the archive.

    Args:
        filepath (str | os.PathLike[str]): Existing local CZI file.
        output_path (str | os.PathLike[str] | None): Destination ending in
            ``.vzip``. Defaults to the source path with its suffix replaced.
            The destination directory must already exist.
        overwrite (bool): Replace an existing archive when True.

    Returns:
        Path: Absolute path of the saved archive.

    Raises:
        FileNotFoundError: The source or destination directory does not exist.
        ValueError: A suffix is unsupported or the source is not a CZI.
        FileExistsError: The destination exists and overwrite is False.
        ImportError: The optional refkv dependency is unavailable.
        OSError: Reading, writing, or publishing the archive fails.

    Notes:
        Unsupported CZI layouts or codecs raise refkv's original exception.
        Archives are not standalone OME-Zarr pixel exports. Sharing or moving
        an archive alone does not make its referenced CZI accessible elsewhere.
    """
    source = Path(filepath).expanduser().resolve()
    if source.suffix.lower() != ".czi":
        raise ValueError(f"Expected a .czi source file: {source}")
    if not source.is_file():
        raise FileNotFoundError(f"CZI source file does not exist: {source}")
    with source.open("rb") as stream:
        if stream.read(16) != b"ZISRAWFILE\x00\x00\x00\x00\x00\x00":
            raise ValueError(f"Source does not have a CZI file header: {source}")
    destination = source.with_suffix(".vzip") if output_path is None else Path(output_path).expanduser().absolute()
    if destination.suffix.lower() != ".vzip":
        raise ValueError(f"Expected a .vzip destination: {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"Destination directory does not exist: {destination.parent}")
    if not overwrite and os.path.lexists(destination):
        raise FileExistsError(f"Archive already exists: {destination}")

    try:
        from refkv.zarrify import zarrify
    except ModuleNotFoundError as error:
        raise ImportError('CZI vzip export requires optional dependencies. Install "czitools[vzip]".') from error

    reference = zarrify(str(source))
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.stem}-",
        suffix=".vzip",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        reference.save(temporary_path)
        if overwrite:
            os.replace(temporary_path, destination)
        else:
            os.link(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    logger.info("Created reference archive %s for CZI %s", destination, source)
    return destination
