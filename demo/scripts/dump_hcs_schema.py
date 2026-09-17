"""EXPERIMENTAL: dump a JSON Schema for CZI metadata documents.

This script generates an experimental JSON Schema (Draft 2020-12) derived from
the ``CziHcsDocument`` dataclass graph. It supports both well-plate and
non-well-plate CZI files; HCS data is optional. The schema is unstable and
expected to change; it is provided for review and iteration only.

Usage:
    python demo/scripts/dump_hcs_schema.py
    python demo/scripts/dump_hcs_schema.py -o docs/schemas/czi-hcs.schema.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from czitools.metadata_tools.hcs_document import CZI_HCS_DOCUMENT_SCHEMA_VERSION
from czitools.metadata_tools.experimental.hcs_schema import write_hcs_json_schema

console = Console()


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("docs/schemas") / f"czi-hcs-{CZI_HCS_DOCUMENT_SCHEMA_VERSION}.experimental.schema.json",
        help="Output JSON Schema path.",
    )
    return parser


def main() -> int:
    """Write the experimental CZI metadata JSON Schema to disk."""
    args = _create_parser().parse_args()
    console.print(
        "[yellow]Warning:[/yellow] The CZI metadata JSON Schema (with optional "
        "HCS data) is experimental and may change without notice."
    )
    output = write_hcs_json_schema(args.output, warn=False)
    console.print(f"[green]Wrote experimental schema:[/green] {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
