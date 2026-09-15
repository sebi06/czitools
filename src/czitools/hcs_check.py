"""Command-line inspector for CZI HCS metadata and stored dimensions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from czitools.metadata_tools import CziMetadata
from czitools.utils.hcs_tools import (
    print_hcs_plate_info,
    print_sample_metadata,
    print_well_fields,
)

console = Console()


def _format_valid_indices(indices: tuple[int, ...]) -> str:
    """Format exact indices without expanding large spatial ranges."""
    if not indices:
        return "[dim]not available[/dim]"
    if len(indices) <= 20:
        return str(list(indices))
    return f"[{indices[0]}, ..., {indices[-1]}] ({len(indices)} indices)"


def print_subblock_dimensions(metadata: CziMetadata) -> None:
    """Display full-resolution dimension sizes and bounds from stored subblocks.

    Args:
        metadata (CziMetadata): CZI metadata containing physical dimensions.
    """
    image = metadata.image
    if image is None:
        console.print("[yellow]No subblock-derived dimensions available.[/yellow]")
        return

    dimensions = (
        ("S", "Scene", image.SizeS),
        ("T", "Time", image.SizeT),
        ("C", "Channel", image.SizeC),
        ("Z", "Z-slice", image.SizeZ),
        ("Y", "Height", image.SizeY),
        ("X", "Width", image.SizeX),
        ("M", "Mosaic", image.SizeM),
        ("R", "Rotation", image.SizeR),
        ("I", "Illumination", image.SizeI),
        ("H", "Phase", image.SizeH),
        ("V", "View", image.SizeV),
        ("B", "Block", image.SizeB),
    )

    table = Table(title="Full-Resolution Subblock Dimensions", style="cyan")
    table.add_column("Dimension", style="bright_cyan", justify="center")
    table.add_column("Meaning")
    table.add_column("Size", style="yellow", justify="right")
    table.add_column("Start", style="green", justify="right")
    table.add_column("End (exclusive)", style="green", justify="right")
    table.add_column("Bounds", style="bright_magenta")
    table.add_column("Valid indices", style="bright_blue")

    for dimension, meaning, size in dimensions:
        valid_indices = image.dimension_indices.get(dimension, ())
        bounds_value = image.dimension_bounds.get(dimension)
        if bounds_value is not None:
            start, end = bounds_value
            bounds = f"[{start}, {end})"
        elif valid_indices:
            start = min(valid_indices)
            end = max(valid_indices) + 1
            bounds = f"[{start}, {end})"
        else:
            start = None
            end = None
            bounds = "[dim]not available[/dim]"

        table.add_row(
            dimension,
            meaning,
            str(size) if size is not None else "[dim]not present[/dim]",
            str(start) if start is not None else "[dim]-[/dim]",
            str(end) if end is not None else "[dim]-[/dim]",
            bounds,
            _format_valid_indices(valid_indices),
        )

    scene_size = (
        f"{image.SizeY_scene} x {image.SizeX_scene} px"
        if image.SizeY_scene is not None and image.SizeX_scene is not None
        else "not available"
    )
    table.caption = f"First stored scene Y x X: {scene_size}"
    console.print(table)


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect CZI HCS metadata and physically stored dimensions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  czi_hcs_check plate.czi
  czi_hcs_check plate.czi --well B5
  czi_hcs_check plate.czi --no-well-table
  czi_hcs_check plate.czi --show-declared
""",
    )
    parser.add_argument(
        "filepath",
        nargs="?",
        type=Path,
        default=None,
        help="Path to the CZI file to inspect.",
    )
    parser.add_argument(
        "-f",
        "--filepath",
        type=Path,
        dest="filepath_flag",
        default=None,
        help="Path to the CZI file to inspect.",
    )
    parser.add_argument(
        "--well",
        type=str,
        default=None,
        help="Well name to inspect, for example B4 or A1.",
    )
    parser.add_argument(
        "--no-well-table",
        action="store_true",
        help="Hide the well summary table.",
    )
    parser.add_argument(
        "--show-declared",
        action="store_true",
        help="Show XML-declared HCS fields instead of only stored fields.",
    )
    return parser


def main() -> int:
    """Run the CZI HCS inspector.

    Returns:
        int: Process exit code.
    """
    args = _create_parser().parse_args()
    filepath = args.filepath_flag if args.filepath_flag else args.filepath
    if filepath is None:
        console.print(
            "[bold red]Error:[/bold red] filepath is required. " "Use -f, --filepath, or a positional path.",
            file=sys.stderr,
        )
        return 1
    if not filepath.exists():
        console.print(f"[bold red]Error:[/bold red] File not found: {filepath}", file=sys.stderr)
        return 1

    try:
        metadata = CziMetadata(
            filepath,
            filter_hcs_to_stored_scenes=not args.show_declared,
        )
        console.print(
            Panel(
                f"[bold]File:[/bold] {filepath}",
                border_style="green",
                style="bold green",
                title="CZI - HCS Inspector",
            )
        )
        print_subblock_dimensions(metadata)
        print_hcs_plate_info(metadata, show_well_table=not args.no_well_table)
        print_sample_metadata(metadata, args.well)
        print_well_fields(metadata, args.well)
        console.print(
            Panel(
                "[bold green]Analysis complete.[/bold green]",
                border_style="green",
                style="dim green",
            )
        )
        return 0
    except Exception as error:
        console.print(
            Panel(
                f"[bold red]{error}[/bold red]",
                title="[bold red]Error Processing File[/bold red]",
                border_style="red",
                style="red",
            )
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
