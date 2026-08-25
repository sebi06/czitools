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

    # Display help:
    python czi_hcs_check.py --help
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from czitools.metadata_tools import CziMetadata
from czitools.utils.hcs_tools import (
    print_hcs_plate_info,
    print_sample_metadata,
    print_well_fields,
)

# Initialize rich console with color support
console = Console()


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
        metadata = CziMetadata(filepath)

        # Display file header with rich styling
        header = Panel(
            f"[bold]File:[/bold] {filepath}",
            border_style="green",
            style="bold green",
            title="📄 CZI - HCS Inspector",
        )
        console.print(header)

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
