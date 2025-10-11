from ome_zarr_utils import define_plate, define_plate_by_well_count, PlateType, PLATE_FORMATS


# Example usage:
if __name__ == "__main__":

    # Method 1: Using PlateType enum (default field_count=1)
    plate_96 = define_plate(PlateType.PLATE_96)
    print(f"96-well plate: {plate_96.name}, Wells: {len(plate_96.wells)}, Fields: {plate_96.field_count}")

    # Method 2: Using well count with custom field count
    plate_384 = define_plate_by_well_count(384, field_count=4)
    print(f"384-well plate: {plate_384.name}, Wells: {len(plate_384.wells)}, Fields: {plate_384.field_count}")

    # Method 3: Using PlateType enum with custom field count
    plate_24 = define_plate(PlateType.PLATE_24, field_count=2)
    print(f"24-well plate: {plate_24.name}, Wells: {len(plate_24.wells)}, Fields: {plate_24.field_count}")

    # Show all available formats
    print("\nAvailable plate formats:")
    for plate_type in PlateType:
        config = plate_type.value
        print(f"  {config.name}: {config.rows}x{config.columns} = {config.total_wells} wells")

    # Example: Create all plate types with varied field counts
    print("\nCreating all plate types with varied field counts:")
    field_counts = {6: 1, 24: 2, 48: 1, 96: 4, 384: 9, 1536: 16}  # Example field counts
    for well_count in PLATE_FORMATS.keys():
        field_count = field_counts[well_count]
        plate = define_plate_by_well_count(well_count, field_count=field_count)
        print(f"  {plate.name}: {len(plate.wells)} wells, {plate.field_count} field(s)")
