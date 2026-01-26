# Linux + Napari + get_planetable() - Quick Guide

## ✅ YES, You Can Use get_planetable() with Napari on Linux!

### The Solution Depends on Your Use Case

#### Use Case 1: Standalone Scripts (Recommended Pattern) ⭐

If you're writing a **standalone script** that starts Napari, use **Sequential Execution**:

Extract the planetable **BEFORE** starting Napari to avoid threading conflicts.

#### Use Case 2: Napari Plugins (napari-czitools)

If czitools is used as a **Napari plugin**, the sequential pattern won't work because
Napari is already running. See [Napari Plugin Section](#for-napari-plugins-napari-czitools) below.

---

## For Standalone Scripts (Sequential Pattern)

```python
from pathlib import Path
from czitools.utils.planetable import get_planetable
from czitools.read_tools import read_tools
import napari

# Your CZI file
czi_file = Path("path/to/your/file.czi")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 1: Extract planetable FIRST (before Napari)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("Extracting planetable...")
df_planetable, csv_path = get_planetable(
    czi_file,
    norm_time=True,      # Normalize timestamps
    save_table=False     # Or True to save CSV
)

print(f"✅ Planetable extracted: {len(df_planetable)} rows")
print(f"   Columns: {list(df_planetable.columns)}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 2: Load image data (thread-safe)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("Loading image...")
array, metadata = read_tools.read_6darray(
    czi_file,
    use_dask=True,       # Lazy loading
    use_xarray=True,     # Labeled dimensions
    chunk_zyx=True       # Optimal chunking
)

print(f"✅ Image loaded: {array.shape}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 3: NOW start Napari (safe - planetable already extracted)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("Starting Napari...")
viewer = napari.Viewer()

# Add image
viewer.add_image(
    array,
    name="CZI Image",
    metadata={'czi_metadata': metadata.info}
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 4: Use planetable data for analysis/visualization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Example: Display timestamps per time point
if 'Time[s]' in df_planetable.columns:
    time_info = df_planetable.groupby('T')['Time[s]'].first()
    print("\nTime points:")
    for t, time_s in time_info.items():
        print(f"  T={t}: {time_s:.2f} seconds")

# Example: Display XY positions
if 'X[micron]' in df_planetable.columns and 'Y[micron]' in df_planetable.columns:
    print("\nStage positions:")
    pos_info = df_planetable.groupby(['T', 'Z', 'C'])[['X[micron]', 'Y[micron]']].first()
    print(pos_info.head())

# Example: Save planetable for later analysis
if csv_path is None:  # If we didn't save earlier
    df_planetable.to_csv("planetable.csv", sep=";", index=True)
    print("\n✅ Planetable saved to: planetable.csv")

print("\n✅ Safe to use Napari - no threading conflicts!")
napari.run()
```

## Why This Works on Linux

### The Problem
- **aicspylibczi** (used by `get_planetable()`) has threading conflicts with **PyQt** (used by Napari)
- When both run simultaneously, crashes can occur on Linux

### The Solution
1. **Extract planetable FIRST** - aicspylibczi runs before PyQt starts
2. **Then start Napari** - PyQt event loop starts after aicspylibczi is done
3. **No conflict** - they never run simultaneously!

### Timeline
```
Before Napari starts:
  ✅ get_planetable() runs (aicspylibczi)
  ✅ read_6darray() runs (pylibCZIrw)

Then Napari starts:
  ✅ PyQt event loop starts
  ✅ Use already-extracted data

No threading conflicts! ✅
```

## What You Get

✅ **Full planetable functionality**
- All subblock metadata
- Timestamps, stage positions, focus positions
- Scene, tile, time, channel, z-plane indices
- Can be saved to CSV

✅ **Full image data**
- Lazy-loaded with dask
- All dimensions preserved
- Thread-safe reading

✅ **Stable Napari on Linux**
- No crashes
- No threading conflicts
- Full GUI functionality

## Common Use Cases

### Use Case 1: Analyze Timing
```python
# Get planetable first
df, _ = get_planetable(czi_file)

# Calculate time intervals
df['Δt'] = df.groupby(['T'])['Time[s]'].diff()

# Load and display
array, _ = read_tools.read_6darray(czi_file, use_dask=True)
viewer = napari.Viewer()
viewer.add_image(array)

# Show timing in console
print(df[['T', 'C', 'Z', 'Time[s]', 'Δt']].head(20))
napari.run()
```

### Use Case 2: Multi-Position Time Series
```python
# Get planetable with all positions
df, _ = get_planetable(czi_file)

# Group by scene (position)
for scene in df['S'].unique():
    scene_data = df[df['S'] == scene]
    print(f"\nScene {scene}:")
    print(f"  Time points: {scene_data['T'].nunique()}")
    print(f"  Channels: {scene_data['C'].nunique()}")
    print(f"  Z-planes: {scene_data['Z'].nunique()}")

# Load specific scene
array, _ = read_tools.read_6darray(czi_file, planes={'S': (scene, scene)}, use_dask=True)

# Display in Napari
viewer = napari.Viewer()
viewer.add_image(array, name=f"Scene {scene}")
napari.run()
```

### Use Case 3: Filter by Metadata
```python
# Get planetable
df, _ = get_planetable(czi_file)

# Filter to specific conditions
df_filtered = df[(df['C'] == 0) & (df['Z'] < 10)]

# Get unique coordinates to load
t_indices = df_filtered['T'].unique()
z_indices = df_filtered['Z'].unique()

# Load filtered data
array, _ = read_tools.read_6darray(
    czi_file,
    planes={
        'T': (t_indices.min(), t_indices.max()),
        'Z': (z_indices.min(), z_indices.max()),
        'C': (0, 0)
    },
    use_dask=True
)

# Display
viewer = napari.Viewer()
viewer.add_image(array, name="Filtered")
napari.run()
```

## Troubleshooting

### Q: Can I call get_planetable() AFTER Napari is running?
**A:** Not recommended on Linux. Extract it before. However, if you need updated planetables, you can:
```python
# Extract multiple planetables before Napari
df1, _ = get_planetable("file1.czi")
df2, _ = get_planetable("file2.czi")

# Then start Napari and use them
viewer = napari.Viewer()
# ... use df1, df2 ...
```

### Q: What if I have many files?
**A:** Extract all planetables first:
```python
planetables = {}
for file in czi_files:
    df, _ = get_planetable(file)
    planetables[file.name] = df

# Now start Napari
viewer = napari.Viewer()
# Use planetables dictionary
```

### Q: Does this work on Windows/macOS too?
**A:** Yes! This pattern works on all platforms. On Windows/macOS you have more flexibility (thread-locked mode usually works), but this sequential pattern is safest everywhere.


## For Napari Plugins (napari-czitools)

### The Challenge

When czitools is used as a **Napari plugin**, the situation is different:

```
User starts Napari
  └─> PyQt event loop running
      └─> User activates plugin
          └─> Plugin code executes
              └─> Can't use "sequential pattern" - Napari already running!
```

The sequential pattern **doesn't work** because Napari is already started when your plugin code runs.

### Solutions for Plugins

#### Option 1: Platform-Aware with Warnings (Recommended)

Detect Linux and warn users, with graceful fallback:

```python
import platform
from czitools.utils.planetable import get_planetable
from czitools.read_tools import read_tools

def load_czi_for_plugin(filepath):
    # Always load image (thread-safe)
    array, metadata = read_tools.read_6darray(filepath, use_dask=True)
    
    # Handle planetable based on platform
    planetable_df = None
    
    if platform.system() == "Linux":
        print("⚠️ Warning: Planetable extraction may crash on Linux")
        print("   If crashes occur, restart with: export CZITOOLS_DISABLE_AICSPYLIBCZI=1")
        
        try:
            planetable_df, _ = get_planetable(filepath)
            print("✅ Planetable extracted (thread-locked)")
        except Exception as e:
            print(f"❌ Planetable failed: {e}")
            planetable_df = None
    else:
        # Windows/macOS - usually safe with thread locks
        planetable_df, _ = get_planetable(filepath)
    
    return array, metadata, planetable_df
```

#### Option 2: Disable on Linux by Default

Skip planetable extraction on Linux unless user explicitly enables:

```python
import platform

def load_czi_for_plugin(filepath, enable_planetable_on_linux=False):
    array, metadata = read_tools.read_6darray(filepath, use_dask=True)
    
    is_linux = platform.system() == "Linux"
    
    if is_linux and not enable_planetable_on_linux:
        print("ℹ️ Planetable disabled on Linux (threading safety)")
        print("   Enable with caution - may cause crashes")
        return array, metadata, None
    
    # Extract planetable
    try:
        planetable_df, _ = get_planetable(filepath)
        return array, metadata, planetable_df
    except Exception as e:
        print(f"Planetable extraction failed: {e}")
        return array, metadata, None
```

#### Option 3: Safe Mode Instructions

Document for users to set environment variable before starting Napari:

```bash
# Linux users: Set this before starting Napari
export CZITOOLS_DISABLE_AICSPYLIBCZI=1
napari
```

Then in plugin:
```python
# get_planetable() will return empty DataFrame in safe mode
# Plugin should handle gracefully
df, _ = get_planetable(filepath)
if df.empty:
    print("Planetable not available (safe mode active)")
```

### Complete Plugin Example

See [demo/scripts/napari_plugin_example.py](../../demo/scripts/napari_plugin_example.py) for a full implementation showing:
- Platform detection
- Graceful error handling
- User warnings
- Optional planetable extraction
- Safe defaults for Linux

### Testing Your Plugin on Linux

1. **Without planetable** (safest):
   ```bash
   export CZITOOLS_DISABLE_AICSPYLIBCZI=1
   napari
   # Activate your plugin
   ```

2. **With planetable** (test carefully):
   ```bash
   napari
   # Activate your plugin
   # Watch for crashes
   ```

3. **If crashes occur**:
   - Restart with `CZITOOLS_DISABLE_AICSPYLIBCZI=1`
   - Disable planetable for Linux users in your plugin
   - Add platform warnings to documentation

### Recommendation for Plugin Developers

**Default behavior for napari-czitools:**
- **Windows/macOS**: Enable planetable (thread locks usually sufficient)
- **Linux**: Disable planetable by default, allow opt-in with warnings

This provides the best balance of:
- ✅ Stability (no crashes out-of-the-box on Linux)
- ✅ Functionality (planetable works on Windows/macOS)
- ✅ Flexibility (Linux users can enable if their system supports it)

---

## Summary: Two Different Scenarios

| Scenario | Pattern | Works? |
|----------|---------|--------|
| **Standalone script** | Sequential (planetable first) | ✅ Yes |
| **Napari plugin** | Thread-locked (Napari already running) | ⚠️ Test on Linux |
| **Plugin + Safe Mode** | Disable aicspylibczi before Napari | ✅ Yes (no planetable) |



**See also:**
- [docs/FINAL_SOLUTION.md](FINAL_SOLUTION.md) - Complete technical details
- [demo/scripts/napari_with_planetable_linux.py](../demo/scripts/napari_with_planetable_linux.py) - Full working example
- [docs/threading_considerations.md](threading_considerations.md) - Threading deep dive
