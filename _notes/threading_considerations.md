# Threading Considerations for czitools

## ⚠️ CRITICAL: Known Issue with aicspylibczi and Napari on Linux

**If you experience crashes when using czitools with Napari running, this is a known issue with the `aicspylibczi` library.**

### Symptoms
- Application crashes when reading CZI files with Napari already running
- Most common on Linux systems
- Related to threading conflicts between aicspylibczi and PyQt event loop

### Solution 1: Process Isolation (RECOMMENDED - Enables ALL Functions)

Run aicspylibczi operations in separate processes to completely isolate them from PyQt:

```python
from czitools.utils.threading_helpers import enable_process_isolation
enable_process_isolation()  # Call FIRST, before any czitools imports

# Now ALL functions work safely with Napari
from czitools.read_tools import read_tools
from czitools.utils.planetable import get_planetable

# Everything works! Including tiles and planetables
tiles, size = read_tools.read_tiles(filepath, scene=0, tile=0)  # ✅
df, path = get_planetable(filepath)  # ✅
array6d, mdata = read_tools.read_6darray(filepath, use_dask=True)  # ✅
```

**How it works:**
- aicspylibczi operations run in separate worker processes
- Complete isolation from PyQt event loop
- No threading conflicts
- Small overhead (~100-500ms per operation)

Or set it system-wide:
```bash
export CZITOOLS_USE_PROCESS_ISOLATION=1
python your_napari_script.py
```

### Solution 2: Legacy Safe Mode (Simpler - Disables Some Functions)

Set the environment variable **before** importing czitools:

```python
import os
os.environ["CZITOOLS_DISABLE_AICSPYLIBCZI"] = "1"

# Now import czitools
from czitools.read_tools import read_tools
from czitools.metadata_tools.czi_metadata import CziMetadata

# Use thread-safe reading methods
array6d, mdata = read_tools.read_6darray(filepath, use_dask=True)
```

Or system-wide:
```bash
export CZITOOLS_DISABLE_AICSPYLIBCZI=1
python your_napari_script.py
```

### Comparison: Process Isolation vs. Legacy Safe Mode

| Feature              | Process Isolation      | Legacy Safe Mode      |
| -------------------- | ---------------------- | --------------------- |
| `read_6darray()`     | ✅ Works                | ✅ Works               |
| `read_stacks()`      | ✅ Works                | ✅ Works               |
| `read_tiles()`       | ✅ **Works!**           | ❌ RuntimeError        |
| `get_planetable()`   | ✅ **Works!**           | ❌ Empty DataFrame     |
| Core metadata        | ✅ Full                 | ✅ Full (minor gaps)   |
| Performance overhead | ~100-500ms/call        | None                  |
| Setup complexity     | One function call      | One env variable      |
| **Recommendation**   | **Use for production** | Use for quick testing |

### Recommended Workflow for Napari Users

```python
import os
os.environ["CZITOOLS_DISABLE_AICSPYLIBCZI"] = "1"

import napari
from czitools.read_tools import read_tools

# This is the SAFE and RECOMMENDED approach
viewer = napari.Viewer()

# Read with dask (thread-safe, efficient)
array6d, mdata = read_tools.read_6darray(
    filepath,
    use_dask=True,      # Lazy loading via pylibCZIrw
    use_xarray=True,    # Labeled dimensions
    chunk_zyx=True      # Optimized chunking
)

# Add to Napari - no threading issues!
for ch in range(mdata.image.SizeC):
    viewer.add_image(
        array6d.sel(C=ch),
        name=f"Channel {ch}",
        scale=[mdata.scale.Z, mdata.scale.Y, mdata.scale.X]
    )

napari.run()
```

## Thread Safety Status

czitools is **generally thread-safe** for reading operations, but there are important considerations when using it in multi-threaded environments like Napari with PyQt on Linux.

## Underlying Library Thread Safety

### libCZI (via pylibCZIrw)
- **Designed for concurrent use**: The core libCZI library is explicitly designed to be thread-safe
- **Read operations**: Can be called from multiple threads concurrently
- **File handle management**: Each read operation opens and closes its own file handle

### Platform-Specific Implementation
- **Windows**: Uses Win32 `ReadFile` with offset for concurrent access without locking
- **Linux**: Uses `fseek`/`fread` with mutex protection OR `pread`-based implementation (both thread-safe)

## Current czitools Implementation

### Safe Patterns
1. **Stateless reading**: Each read operation is independent
2. **Context managers**: Proper file handle cleanup with `with pyczi.open_czi(...) as czidoc`
3. **No shared mutable state**: Functions don't rely on global variables

### Potential Concerns
1. **File handle limits**: Multiple concurrent reads open multiple file handles
2. **Metadata caching**: No caching mechanism for repeated metadata reads
3. **aicspylibczi usage**: Resource warnings due to lack of explicit `close()` method

## Best Practices for Multi-threaded Usage

### 1. Reuse Metadata Objects
```python
# GOOD: Create metadata once, reuse
mdata = CziMetadata(filepath)
for plane in planes:
    # Use mdata in multiple threads
    pass

# AVOID: Creating metadata repeatedly in threads
def worker():
    mdata = CziMetadata(filepath)  # Opens file each time
```

### 2. Use Thread Pools with Reasonable Limits
```python
from concurrent.futures import ThreadPoolExecutor
from czitools.read_tools import read_tools

# Limit concurrent file access
MAX_WORKERS = 4  # Adjust based on system

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(read_tools.read_2dplane, filepath, ...) 
               for ... in items]
```

### 3. Consider Process-based Parallelism for Heavy Loads
```python
from multiprocessing import Pool

# For CPU-intensive operations, use processes instead of threads
# This avoids GIL limitations and file handle contention
with Pool(processes=4) as pool:
    results = pool.map(process_czi_file, file_list)
```

### 4. Use Dask for Lazy Reading
```python
# RECOMMENDED for Napari integration
array6d, mdata = read_tools.read_6darray(
    filepath,
    use_dask=True,  # Enables lazy evaluation
    use_xarray=True
)

# Dask handles threading internally and efficiently
```

## Napari Integration Considerations

### PyQt Event Loop
When using czitools with Napari (which uses PyQt):

1. **Avoid blocking the GUI thread**:
```python
import napari
from qtpy.QtCore import QThread, Signal

class CziReaderThread(QThread):
    finished = Signal(object)
    
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
    
    def run(self):
        from czitools.read_tools import read_tools
        array6d, mdata = read_tools.read_6darray(
            self.filepath,
            use_dask=True
        )
        self.finished.emit((array6d, mdata))

# Usage
viewer = napari.Viewer()
thread = CziReaderThread(filepath)
thread.finished.connect(lambda data: add_to_napari(viewer, data))
thread.start()
```

2. **Use napari's threading utilities**:
```python
from napari.qt.threading import thread_worker

@thread_worker
def read_czi_worker(filepath):
    from czitools.read_tools import read_tools
    return read_tools.read_6darray(filepath, use_dask=True)

# This handles threading automatically
worker = read_czi_worker(filepath)
worker.returned.connect(lambda data: add_to_viewer(data))
worker.start()
```

### Dask Integration with Napari
Dask arrays work excellently with Napari:
```python
# This is the RECOMMENDED approach
array6d, mdata = read_tools.read_6darray(
    filepath,
    use_dask=True,      # Lazy loading
    use_xarray=True,    # Labeled dimensions
    chunk_zyx=True      # Optimize chunking
)

# Napari handles dask arrays efficiently
viewer = napari.Viewer()
for ch in range(mdata.image.SizeC):
    viewer.add_image(
        array6d.sel(C=ch),
        name=f"Channel {ch}",
        # Dask delays computation until needed
    )
```

## Known Limitations

### 1. aicspylibczi File Handles
- **Issue**: `aicspylibczi.CziFile` doesn't provide a `close()` method
- **Impact**: May see ResourceWarnings in multi-threaded contexts
- **Mitigation**: Explicit cleanup is implemented with `del` and `gc.collect()`

### 2. Concurrent Metadata Reads
- **Current**: Each `CziMetadata` instance opens the file independently
- **Recommendation**: Cache metadata objects when reading from the same file

### 3. OS File Descriptor Limits
- **Linux default**: Typically 1024 open files per process
- **Mitigation**: Use thread pools with limited workers

## Performance Optimization

### For High-Throughput Applications

```python
import threading
from czitools.metadata_tools.czi_metadata import CziMetadata

# Thread-local storage for file handles (advanced)
thread_local = threading.local()

def get_cached_metadata(filepath):
    if not hasattr(thread_local, 'metadata_cache'):
        thread_local.metadata_cache = {}
    
    if filepath not in thread_local.metadata_cache:
        thread_local.metadata_cache[filepath] = CziMetadata(filepath)
    
    return thread_local.metadata_cache[filepath]
```

## Testing Thread Safety

Example test for concurrent access:
```python
import concurrent.futures
from czitools.read_tools import read_tools

def test_concurrent_reads(filepath):
    def read_plane(s, t, c, z):
        array, _ = read_tools.read_6darray(
            filepath,
            planes={'T': (t, t), 'Z': (z, z), 'C': (c, c)}
        )
        return array
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(read_plane, 0, t, 0, 0)
            for t in range(10)
        ]
        results = [f.result() for f in futures]
    
    assert all(r is not None for r in results)
```

## Recommendations Summary

1. ✅ **Use dask for lazy loading** - Best for Napari integration
2. ✅ **Limit concurrent file access** - Use thread pools with reasonable limits
3. ✅ **Cache metadata objects** - Avoid repeated file opens
4. ✅ **Use napari's threading utilities** - Let napari manage threads
5. ⚠️ **Monitor file handles** - Check system limits for high-concurrency scenarios
6. ⚠️ **Test on target platform** - Linux file I/O characteristics differ from Windows

## Conclusion

**czitools reading is thread-safe** due to the underlying libCZI design. For optimal performance in Napari with PyQt on Linux:
- Use dask-based lazy reading (`use_dask=True`)
- Leverage napari's built-in threading support
- Limit concurrent operations to avoid resource exhaustion
- Cache metadata objects when reading the same file multiple times

The library should work without issues in multi-threaded environments, but following these best practices will ensure optimal performance and resource utilization.
