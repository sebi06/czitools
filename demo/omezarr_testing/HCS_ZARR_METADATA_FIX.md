# HCS-ZARR Metadata Structure Issue and Fix

## Problem Summary

The error `IndexError: list index out of range` occurs when trying to access:
```python
row = plate.metadata.rows[well_meta.rowIndex].name
col = plate.metadata.columns[well_meta.columnIndex].name
```

## Root Cause

There are **two different approaches** to creating HCS-ZARR files, which result in different metadata structures:

### Approach 1: `ome-zarr-py` (Incomplete Metadata)
Used in `convert_czi_to_hcs_zarr()`:
```python
from ome_zarr.writer import write_plate_metadata

write_plate_metadata(root, row_names, col_names, well_paths)
```

**Problem:** This function writes basic plate metadata but **does NOT populate** the `rows` and `columns` arrays in the metadata structure. When the file is read back:
- `plate.metadata.wells` ✅ Contains well information with correct `rowIndex` and `columnIndex`
- `plate.metadata.rows` ❌ Empty list `[]`
- `plate.metadata.columns` ❌ Empty list `[]`

**Result:** Accessing `plate.metadata.rows[1]` causes `IndexError` because the list is empty.

### Approach 2: `ngff-zarr` (Complete Metadata)
Used in `convert_czi_to_hcsplate()`:
```python
from ngff_zarr.v04.zarr_metadata import Plate, PlateColumn, PlateRow

columns = [PlateColumn(name=str(col)) for col in sorted(col_names, key=int)]
rows = [PlateRow(name=row) for row in sorted(row_names)]
plate = Plate(columns=columns, rows=rows, wells=wells, ...)
```

**Success:** This approach explicitly creates `PlateRow` and `PlateColumn` objects:
- `plate.metadata.wells` ✅ Contains well information
- `plate.metadata.rows` ✅ Contains `[PlateRow(name='A'), PlateRow(name='B'), ...]`
- `plate.metadata.columns` ✅ Contains `[PlateColumn(name='1'), PlateColumn(name='2'), ...]`

**Result:** Index-based access works correctly.

## The Fix

### Option 1: Update `convert_czi_to_hcs_zarr()` (Recommended)

Add proper row/column metadata to the ZARR attributes:

```python
# Create PlateRow and PlateColumn objects
columns_metadata = [PlateColumn(name=str(col)) for col in sorted(col_names, key=int)]
rows_metadata = [PlateRow(name=row) for row in sorted(row_names)]

# Write plate metadata
write_plate_metadata(root, row_names, col_names, well_paths)

# Store rows and columns in metadata for compatibility
plate_attrs = root.attrs.asdict()
plate_attrs['rows'] = [{'name': r.name} for r in rows_metadata]
plate_attrs['columns'] = [{'name': c.name} for c in columns_metadata]
root.attrs.update(plate_attrs)
```

### Option 2: Use Robust Access Pattern (Workaround)

Always extract row/column from the well path instead of using indices:

```python
# Instead of:
# row = plate.metadata.rows[well_meta.rowIndex].name
# col = plate.metadata.columns[well_meta.columnIndex].name

# Use:
row, col = well_meta.path.split('/')
```

**Advantages:**
- ✅ Always works regardless of metadata structure
- ✅ `well_meta.path` is always correctly formatted (e.g., "B/4")
- ✅ No dependency on rows/columns lists being populated
- ✅ More robust and simpler

### Option 3: Use `convert_czi_to_hcsplate()` Instead

Switch to using the `ngff-zarr` based function which creates complete metadata:

```python
# Replace:
zarr_output_path = convert_czi_to_hcs_zarr(filepath, overwrite=True)

# With:
zarr_output_path = convert_czi_to_hcsplate(filepath, plate_name="My Plate", overwrite=True)
```

## Recommended Solution

**Use Option 2** (path-based access) as it's the most robust:

```python
for well_meta in plate.metadata.wells:
    # Extract row and column from path (format: "B/4")
    row, col = well_meta.path.split('/')
    
    # Get the well object
    well = plate.get_well(row, col)
    
    # Process well...
```

This approach:
- Works with both metadata structures
- Is simpler and more maintainable
- Avoids the entire indexing issue
- The `path` is the source of truth

## Why rowIndex and columnIndex Exist

The `rowIndex` and `columnIndex` fields in `PlateWell` are meant to be indices into the `plate.metadata.rows` and `plate.metadata.columns` arrays. They're useful for:
- Quick lookups when the arrays are populated
- Numerical operations on plate coordinates
- Compatibility with other tools

However, they're **optional** in the spec and **should not be relied upon** unless you control the entire metadata creation process.

## Testing

To verify if a ZARR file has complete metadata:

```python
import ngff_zarr as nz

plate = nz.from_hcs_zarr(zarr_path, validate=True)

print(f"Rows: {len(plate.metadata.rows)}")  # Should be > 0
print(f"Columns: {len(plate.metadata.columns)}")  # Should be > 0
print(f"Wells: {len(plate.metadata.wells)}")  # Should match actual wells

if plate.metadata.rows:
    print(f"Row names: {[r.name for r in plate.metadata.rows]}")
if plate.metadata.columns:
    print(f"Column names: {[c.name for c in plate.metadata.columns]}")
```

## Conclusion

The safest and most reliable approach is to **always extract row/column from the well path** rather than using index-based lookup. This makes your code compatible with any HCS-ZARR file, regardless of how it was created.
