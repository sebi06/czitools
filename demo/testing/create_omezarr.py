import os
import shutil
import numpy as np
import zarr
from ome_zarr.io import parse_url
from ome_zarr.writer import write_multiscales_metadata
from ome_zarr.format import FormatV04

# define the output path
path = "file.ome.zarr"

# remove existing Zarr directory
if os.path.exists(path):
    shutil.rmtree(path)

# create a 5D array (T, C, Z, Y, X)
size_t = 2  # 2 time points
size_c = 1  # 1 channel
size_z = 10  # 10 Z slices
size_yx = 1024  # 1024 x 1024 pixels

# simulate data
data = np.random.randint(
    0, 255, (size_t, size_c, size_z, size_yx, size_yx), dtype=np.uint16
)

# store in OME-Zarr format
store = parse_url(path, mode="w").store
root = zarr.group(store=store)

# write image data
root.create_dataset("0", data=data, chunks=(1, 1, 1, 512, 512), dtype=np.uint8)

# define voxel size parameters
pixel_size_um = 0.1  # Microns per pixel (XY)
z_step_um = 5  # Microns per Z slice
t_spacing_s = 20  # Time interval in seconds

# define metadata
datasets = [
    {
        "path": "0",
        "coordinateTransformations": [
            {
                "type": "scale",
                "scale": [
                    t_spacing_s,  # Time step (seconds)
                    1.0,  # Channel
                    z_step_um,  # Z-spacing (micrometers)
                    pixel_size_um,  # Pixel size Y (micrometers)
                    pixel_size_um,  # Pixel size X (micrometers)
                ],
            }
        ],
    }
]

# axes with units
axes = [
    {"name": "t", "type": "time", "unit": "microsecond"},
    {"name": "c", "type": "channel"},
    {"name": "z", "type": "space", "unit": "micrometer"},
    {"name": "y", "type": "space", "unit": "micrometer"},
    {"name": "x", "type": "space", "unit": "micrometer"},
]

# write OME-Zarr metadata
write_multiscales_metadata(
    group=root,
    datasets=datasets,
    fmt=FormatV04(),
    axes=axes,
    name="image",
)
