"""Create a reference-backed CZI archive without importing a GUI.

Run from the repository root in an activated environment:
    python demo/scripts/create_czi_vzip.py image.czi
    python demo/scripts/create_czi_vzip.py image.czi -o archives/image.vzip
    python demo/scripts/create_czi_vzip.py image.czi --overwrite

Install czitools[vzip] first. The destination directory must exist. The saved
archive still requires the unchanged CZI at its original referenced location.
"""

import argparse
from pathlib import Path

from czitools.export_tools import create_czi_vzip


def main(argv: list[str] | None = None) -> int:
    """Export one CZI using the same API a future napari worker can call.

    Args:
        argv (list[str] | None): Arguments, or None for command-line arguments.

    Returns:
        int: Zero after a successful export.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Existing local .czi file")
    parser.add_argument("-o", "--output", type=Path, help="Destination .vzip (default: beside the CZI)")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing archive")
    args = parser.parse_args(argv)
    try:
        archive = create_czi_vzip(args.source, args.output, overwrite=args.overwrite)
    except (OSError, ValueError, ImportError) as error:
        parser.exit(2, f"Export failed: {error}\n")
    print(f"Saved: {archive}")
    print("Keep the original CZI unchanged at its referenced location.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())