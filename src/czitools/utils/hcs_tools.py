"""HCS plate metadata display utilities using rich formatting.

Provides rich-formatted output for High-Content Screening (HCS) plate information,
including plate hierarchy, sample metadata, and detailed field information.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from czitools.metadata_tools import CziMetadata

# Initialize rich console with color support
console = Console()


def print_section_header(title: str) -> None:
    """Print a formatted section header with rich styling.

    Args:
        title: The title of the section to display.
    """
    panel = Panel(
        title,
        border_style="cyan",
        style="bold cyan",
    )
    console.print(panel)


def print_hcs_plate_info(metadata: CziMetadata, show_well_table: bool = True) -> None:
    """Print High-Content Screening (HCS) plate information with rich formatting.

    Displays plate hierarchy, dimensions, well count, and optionally a summary table of
    all wells with their positions and field counts.

    Args:
        metadata: CziMetadata object containing file information.
        show_well_table: Whether to display the well summary table. Defaults to True.
            Set to False to hide the table for large plates.
    """
    hcs = metadata.hcs
    if hcs is None:
        console.print("\n[yellow]⚠ No HCS plate hierarchy detected in this file.[/yellow]")
        return

    print_section_header("🔬 High-Content Screening (HCS) Plate Information")

    detected_text = Text(str(metadata.hcs_status.detected), style="green bold")
    console.print(f"HCS Detected: {detected_text}")
    console.print(f"Reason: [dim]{metadata.hcs_status.reason}[/dim]")

    plate = hcs
    console.print()
    console.print(f"[bold magenta]Plate ID:[/bold magenta] {plate.id}")
    console.print(f"[bold magenta]Plate Name:[/bold magenta] {plate.name}")
    console.print(f"[bold magenta]Schema Version:[/bold magenta] {plate.schema_version}")
    console.print(
        f"[bold magenta]Dimensions:[/bold magenta] {plate.declared_rows} rows × {plate.declared_columns} columns (declared)"
    )
    console.print(f"[bold magenta]Row Indices:[/bold magenta] {plate.observed_row_indices}")
    console.print(f"[bold magenta]Column Indices:[/bold magenta] {plate.observed_column_indices}")
    console.print()
    console.print(f"[bold yellow]Total Wells:[/bold yellow] {len(plate.wells)}")
    console.print(f"[bold yellow]Total Fields:[/bold yellow] {sum(len(well.fields) for well in plate.wells)}")

    # Display wells in a rich formatted table (optional)
    if show_well_table:
        table = Table(title="Well Summary", style="cyan")
        table.add_column("Well", style="bright_cyan", no_wrap=False)
        table.add_column("Path", style="dim")
        table.add_column("CZI Index", style="magenta")
        table.add_column("Normalized", style="magenta")
        table.add_column("Fields", style="yellow", justify="right")

        for well in plate.wells:
            table.add_row(
                well.canonical_name,
                well.canonical_path,
                f"({well.source_row_index},{well.source_column_index})",
                f"({well.row_index},{well.column_index})",
                str(len(well.fields)),
            )

        console.print(table)


def print_sample_metadata(metadata: CziMetadata, well_name: str | None = None) -> None:
    """Print sample metadata and first scene details with rich formatting.

    Displays scene count, unique wells, per-scene collections, and coordinates of the
    first imaging scene. If a well name is provided, shows the first scene from that
    well; otherwise shows the overall first scene.

    Args:
        metadata: CziMetadata object containing file information.
        well_name: Optional well name (e.g., 'B4', 'A1'). If provided, shows the first
            scene from this well. If None, shows the overall first scene.
    """
    sample = metadata.sample
    if sample is None:
        console.print("[yellow]⚠ No sample metadata available.[/yellow]")
        return

    print_section_header("📊 Sample Metadata")

    # Create a table for basic sample info
    info_table = Table(show_header=False, box=None, padding=(0, 2))
    info_table.add_row("[bold]Scene Count:[/bold]", f"[green]{sample.scene_count}[/green]")
    info_table.add_row("[bold]Unique Wells:[/bold]", f"[green]{sample.well_unique_number}[/green]")
    info_table.add_row("[bold]Fields per Well:[/bold]", f"[green]{sample.multipos_per_well}[/green]")
    console.print(info_table)

    # Per-scene collection summary
    per_scene_lengths = {
        "well names": len(sample.well_array_names),
        "well indices": len(sample.well_indices),
        "position names": len(sample.well_position_names),
        "row indices": len(sample.well_rowID),
        "column indices": len(sample.well_colID),
        "field center X": len(sample.field_centerX),
        "field center Y": len(sample.field_centerY),
        "region IDs": len(sample.well_region_ids),
    }

    console.print(f"\n[bold cyan]Per-Scene Collections[/bold cyan] [dim]({len(per_scene_lengths)} entries)[/dim]")
    coll_table = Table(show_header=False, box=None, padding=(0, 2))
    for label, length in per_scene_lengths.items():
        coll_table.add_row(f"  [dim]{label:<20}[/dim]", f"[yellow]{length}[/yellow]")
    console.print(coll_table)

    # First scene details
    if sample.scene_count:
        # Determine which scene index to display
        scene_index = 0
        scene_label = "First Scene Details"

        if well_name is not None:
            # Find first scene for the specified well
            try:
                # Normalize well name to match format in well_array_names
                hcs = metadata.hcs
                normalized_well = None
                if hcs is not None:
                    for well in hcs.wells:
                        if well.canonical_name.upper() == well_name.upper():
                            normalized_well = well.canonical_name
                            break

                if normalized_well is None:
                    # If not found in plate, try direct match
                    normalized_well = well_name.upper()

                # Find first scene for this well
                for idx, scene_well in enumerate(sample.well_array_names):
                    if scene_well and scene_well.upper() == normalized_well:
                        scene_index = idx
                        scene_label = f"First Scene Details (Well {normalized_well})"
                        break
            except (AttributeError, IndexError):
                # Fallback to first scene if well lookup fails
                pass

        print_section_header(f"🎬 {scene_label}")
        scene_table = Table(show_header=False, box=None, padding=(0, 2))
        scene_table.add_row(
            "[bold]Well:[/bold]", f"[green]{sample.well_array_names[scene_index] or '<missing>'}[/green]"
        )
        scene_table.add_row("[bold]Region ID:[/bold]", f"[magenta]{sample.well_region_ids[scene_index]}[/magenta]")
        scene_table.add_row(
            "[bold]Field Center:[/bold]",
            f"[cyan]({sample.field_centerX[scene_index]}, {sample.field_centerY[scene_index]}) µm[/cyan]",
        )
        scene_table.add_row(
            "[bold]Stage Position:[/bold]",
            f"[cyan]({sample.scene_stageX[scene_index]}, {sample.scene_stageY[scene_index]})[/cyan]",
        )
        console.print(scene_table)


def print_well_fields(metadata: CziMetadata, well_name: str | None) -> None:
    """Print detailed field information for a specific well with rich formatting.

    Displays all imaging fields in the specified well, including their local indices,
    scene indices, region IDs, and spatial coordinates. Well names are normalized
    (e.g., 'B04' resolves to 'B4').

    Args:
        metadata: CziMetadata object containing file information.
        well_name: Name of the well to inspect (e.g., 'B4', 'A1'). If None, uses the
            first well in the plate.

    Raises:
        ValueError: If the specified well is not found in the plate.
    """
    hcs = metadata.hcs
    if hcs is None:
        console.print("[yellow]⚠ Cannot display well fields: No HCS plate hierarchy detected.[/yellow]")
        return

    plate = hcs

    # Use first well if not specified
    if well_name is None:
        if not plate.wells:
            console.print("[yellow]⚠ No wells found in this file.[/yellow]")
            return
        well = plate.wells[0]
        well_name = well.canonical_name
        print_section_header(f"🔎 Well Fields (default: first well {well_name})")
    else:
        try:
            well = plate.get_well(well_name)
            print_section_header("🔎 Well Fields")
        except KeyError:
            console.print(f"\n[red]❌ Error: Well {well_name!r} not found in plate.[/red]")
            available = [w.canonical_name for w in plate.wells]
            console.print(f"[yellow]Available wells: {', '.join(available)}[/yellow]")
            return

    # Create fields table
    fields_table = Table(title=f"Fields in well {well_name!r}", style="green")
    fields_table.add_column("Local", style="bright_yellow", justify="center")
    fields_table.add_column("Scene", style="bright_cyan", justify="center")
    fields_table.add_column("ID", style="magenta", justify="center")
    fields_table.add_column("Region", style="bright_blue", justify="center")
    fields_table.add_column("Center X", style="green", justify="right")
    fields_table.add_column("Center Y", style="green", justify="right")
    fields_table.add_column("Unit", style="dim")

    if not well.fields:
        console.print("[yellow]⚠ No fields found in this well.[/yellow]")
        return

    for field in well.fields:
        fields_table.add_row(
            str(field.field_index),
            str(field.scene_index),
            str(field.id),
            str(field.region_id),
            f"{field.scene_center_x:.2f}",
            f"{field.scene_center_y:.2f}",
            str(field.position_unit),
        )

    console.print(fields_table)
