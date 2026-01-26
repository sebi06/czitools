# Quick Fix for Napari Crashes with czitools on Linux

## Problem
Getting crashes when using czitools with Napari on Linux? This is a known threading issue with the `aicspylibczi` library.

## Solution (Choose One)

### Option 1: Set Environment Variable (Easiest)

**At the very top of your script:**
```python
import os
os.environ["CZITOOLS_DISABLE_AICSPYLIBCZI"] = "1"

# Now import everything else
import napari
from czitools.read_tools import read_tools

# Read CZI - this is now thread-safe
array6d, mdata = read_tools.read_6darray(
    "your_file.czi",
    use_dask=True,
    use_xarray=True
)

# Use with Napari
viewer = napari.Viewer()
viewer.add_image(array6d)
napari.run()
```

### Option 2: Use Helper Function

```python
from czitools.utils.napari_helpers import enable_napari_safe_mode
enable_napari_safe_mode()  # Must be FIRST!

import napari
from czitools.read_tools import read_tools

array6d, mdata = read_tools.read_6darray("your_file.czi", use_dask=True)
viewer = napari.Viewer()
viewer.add_image(array6d)
napari.run()
```

### Option 3: Shell Environment (System-wide)

```bash
export CZITOOLS_DISABLE_AICSPYLIBCZI=1
python your_napari_script.py
```

## What Still Works?

With thread-safe mode enabled:
- ✅ `read_6darray()` - **RECOMMENDED** for Napari
- ✅ `read_stacks()` - Also safe
- ✅ `CziMetadata` - Core metadata still available
- ✅ All metadata classes work
- ❌ `read_tiles()` - Not available (raises error)
- ❌ `get_planetable()` - Not available (returns empty)

## Complete Working Example

See `demo/scripts/napari_safe_loading.py` for a full working example.

## Why Does This Happen?

The `aicspylibczi` library has threading conflicts with PyQt (which Napari uses). The thread-safe mode disables `aicspylibczi` and uses only `pylibCZIrw`, which is confirmed thread-safe.

## More Information

- Full documentation: [docs/threading_considerations.md](threading_considerations.md)
- Example script: [demo/scripts/napari_safe_loading.py](../demo/scripts/napari_safe_loading.py)
- Helper utilities: `czitools.utils.napari_helpers`
