"""Export CZI well-plate and HCS metadata as a portable JSON document."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

from czitools.metadata_tools import CziHcsDocument, extract_hcs_document

console = Console()


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filepath", type=Path, help="Input CZI file.")
    parser.add_argument("-o", "--output", type=Path, help="Output JSON path.")
    parser.add_argument(
        "--enrich-positions",
        action="store_true",
        help="Scan subblocks for stage and focus positions.",
    )
    parser.add_argument(
        "--checksum",
        action="store_true",
        help="Include the source file SHA-256 checksum.",
    )
    parser.add_argument(
        "--redact-user",
        action="store_true",
        help="Omit the acquisition user name.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser


def main() -> int:
    """Export one CZI and print a machine-readable document summary."""
    args = _create_parser().parse_args()
    filepath: Path = args.filepath
    if not filepath.exists():
        console.print(f"[red]Error:[/red] File not found: {filepath}")
        return 1

    output: Path = args.output or filepath.with_suffix(".hcs.json")
    if output.exists() and not args.force:
        console.print(f"[red]Error:[/red] Output exists (use --force to overwrite): {output}")
        return 1

    try:
        document = extract_hcs_document(
            filepath,
            enrich_positions=args.enrich_positions,
            include_checksum=args.checksum,
            redact_user=args.redact_user,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        document.write_json(output)
        restored = CziHcsDocument.read_json(output)
    except Exception as error:  # noqa: BLE001 - surface any failure to the CLI user
        console.print(f"[red]Error processing {filepath}:[/red] {error}")
        return 1

    table = Table(title="CZI HCS Metadata Export")
    table.add_column("Property")
    table.add_column("Value", style="cyan")
    table.add_row("Output", str(output.resolve()))
    table.add_row("Schema", restored.schema_version)
    table.add_row("HCS detected", str(restored.hcs.detected))
    table.add_row("Stored wells", str(len(list(restored.iter_well_rows()))))
    table.add_row("Stored fields", str(restored.hcs.stored_field_count))
    table.add_row("Channels", str(len(restored.channels)))
    table.add_row("Quality", restored.quality.status)
    console.print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
