"""CZI HCS Plate Information Viewer.

A command-line tool to inspect and display metadata from CZI (Carl Zeiss Image) files,
including High-Content Screening (HCS) plate information, sample metadata, and detailed
field information for specific wells.

Usage:
    python czi_hcs_check.py [-f FILEPATH] [--well WELL]
    python czi_hcs_check.py FILEPATH [--well WELL]  (positional)

Examples:
    # Display all plate information and first well details (positional):
    python czi_hcs_check.py "C:/path/to/image.czi"

    # Display using -f flag:
    python czi_hcs_check.py -f "C:/path/to/image.czi"

    # Display using --filepath flag:
    python czi_hcs_check.py --filepath "C:/path/to/image.czi"

    # Display information for a specific well:
    python czi_hcs_check.py -f "C:/path/to/image.czi" --well B5

    # Display the complete XML-declared acquisition model:
    python czi_hcs_check.py -f "C:/path/to/image.czi" --show-declared

    # Display help:
    python czi_hcs_check.py --help
"""

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

# Initialize rich console with color support
console = Console()


def print_subblock_dimensions(metadata: CziMetadata) -> None:
    """Display full-resolution dimension sizes derived from stored subblocks.

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
    for dimension, meaning, size in dimensions:
        value = str(size) if size is not None else "[dim]not present[/dim]"
        table.add_row(dimension, meaning, value)

    scene_size = (
        f"{image.SizeY_scene} × {image.SizeX_scene} px"
        if image.SizeY_scene is not None and image.SizeX_scene is not None
        else "not available"
    )
    table.caption = f"First stored scene Y × X: {scene_size}"
    console.print(table)


def main() -> int:
    """Main entry point for the CZI HCS plate information viewer.

    Parses command-line arguments and displays CZI metadata for the specified file
    and optional well.

    Returns:
        int: Exit code (0 for success, 1 for error).
    """
    parser = argparse.ArgumentParser(
        description="Inspect and display metadata from CZI High-Content Screening files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Display all plate information and first well details:
  python czi_hcs_check.py "C:/path/to/image.czi"

  # Display information for a specific well:
  python czi_hcs_check.py "C:/path/to/image.czi" --well B5

  # Display information with a different well format:
  python czi_hcs_check.py "C:/path/to/image.czi" --well "B04"

    # Display the complete XML-declared acquisition model:
    python czi_hcs_check.py "C:/path/to/image.czi" --show-declared
        """,
    )

    parser.add_argument(
        "filepath",
        nargs="?",
        type=Path,
        default=None,
        help="Path to the CZI file to inspect (positional argument)",
    )

    parser.add_argument(
        "-f",
        "--filepath",
        type=Path,
        dest="filepath_flag",
        default=None,
        help="Path to the CZI file to inspect (can use -f or --filepath flag)",
    )

    parser.add_argument(
        "--well",
        type=str,
        default=None,
        help="Well name to inspect (e.g., 'B4', 'A1'). If not provided, uses the first well. "
        "Names are normalized (e.g., 'B04' becomes 'B4').",
    )

    parser.add_argument(
        "--no-well-table",
        action="store_true",
        help="Hide the well summary table (useful for large plates with many wells).",
    )

    parser.add_argument(
        "--show-declared",
        action="store_true",
        help="Show all XML-declared HCS fields instead of only fields backed " "by physical layer-0 subblocks.",
    )

    args = parser.parse_args()

    # Resolve filepath from either positional or flag argument
    filepath = args.filepath_flag if args.filepath_flag else args.filepath
    if filepath is None:
        print("❌ Error: filepath is required. Use -f, --filepath, or provide as positional argument.", file=sys.stderr)
        return 1

    # Validate file exists
    if not filepath.exists():
        print(f"❌ Error: File not found: {filepath}", file=sys.stderr)
        return 1

    try:
        # Read CZI metadata
        metadata = CziMetadata(
            filepath,
            filter_hcs_to_stored_scenes=not args.show_declared,
        )

        # Display file header with rich styling
        header = Panel(
            f"[bold]File:[/bold] {filepath}",
            border_style="green",
            style="bold green",
            title="📄 CZI - HCS Inspector",
        )
        console.print(header)

        # Display physical dimensions from full-resolution subblocks
        print_subblock_dimensions(metadata)

        # Display plate information
        print_hcs_plate_info(metadata, show_well_table=not args.no_well_table)

        # Display sample metadata with well-specific first scene if provided
        print_sample_metadata(metadata, args.well)

        # Display well-specific field information
        print_well_fields(metadata, args.well)

        # Display footer
        footer = Panel(
            "[bold green]✓ Analysis complete![/bold green]",
            border_style="green",
            style="dim green",
        )
        console.print(footer)
        return 0

    except Exception as e:
        error_panel = Panel(
            f"[bold red]{str(e)}[/bold red]",
            title="[bold red]Error Processing File[/bold red]",
            border_style="red",
            style="red",
        )
        console.print(error_panel)
        return 1


if __name__ == "__main__":
    sys.exit(main())
