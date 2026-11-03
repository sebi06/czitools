"""Open an existing CZI-derived vzip in napari without zarrifying again.

Run in an activated environment with czitools[vzip] and napari installed:
    python demo/scripts/view_czi_vzip_napari.py image.vzip

Each image series is opened lazily at its first resolution level. Channels
become separate layers, while time, Z, and other dimensions remain navigable.
The original CZI must still be accessible at the location the archive records.
No pixels are copied and no archive is rewritten by this example.
"""

import argparse
from pathlib import Path
from typing import Any, cast


def main(argv: list[str] | None = None) -> int:
    """Display an archive using its OME axes and channel display metadata.

    Args:
        argv (list[str] | None): Arguments, or None for command-line arguments.

    Returns:
        int: Zero after the viewer closes.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Existing CZI-derived .vzip file")
    args = parser.parse_args(argv)
    if not args.archive.is_file():
        parser.error(f"Archive does not exist: {args.archive}")

    import dask.array as da
    import napari
    import numpy as np
    import refkv.zarr
    import zarr

    store = refkv.zarr.open(str(args.archive.resolve()))
    root = zarr.open_group(store, mode="r")
    viewer = napari.Viewer(title=f"{args.archive.name} - reference-backed CZI")
    try:
        for series_name, series in root.groups():
            ome = cast(dict[str, Any], series.attrs.get("ome", {}))
            for multiscale in ome.get("multiscales", []):
                dataset = multiscale["datasets"][0]
                image = series.get_array(dataset["path"])
                axes = tuple(axis["name"].lower() for axis in multiscale["axes"])
                if len(axes) != image.ndim or len(set(axes)) != len(axes) or not {"y", "x"}.issubset(axes):
                    raise ValueError(f"Unsupported image axes for series {series_name}: {axes}")
                display_axes = tuple(axis for axis in axes if axis not in ("y", "x")) + ("y", "x")
                data = da.from_zarr(image).transpose(tuple(axes.index(axis) for axis in display_axes))
                channel_axis = display_axes.index("c") if "c" in display_axes else None
                layer_axes = tuple(axis for axis in display_axes if axis != "c")
                scale = next(
                    (
                        transform["scale"]
                        for transform in dataset.get("coordinateTransformations", [])
                        if transform["type"] == "scale"
                    ),
                    [1.0] * image.ndim,
                )
                channels = ome.get("omero", {}).get("channels", [])
                count = image.shape[axes.index("c")] if channel_axis is not None else 1
                fallback = (
                    (float(np.iinfo(image.dtype).min), float(np.iinfo(image.dtype).max))
                    if np.issubdtype(image.dtype, np.integer) else (0.0, 1.0)
                )
                names = []
                limits = []
                for index in range(count):
                    channel = channels[index] if index < len(channels) else {}
                    names.append(f"{series_name}: {channel.get('label', f'Channel {index}')}")
                    window = channel.get("window", {})
                    limits.append((window.get("start", fallback[0]), window.get("end", fallback[1])))
                viewer.add_image(
                    data,
                    channel_axis=channel_axis,
                    name=names if channel_axis is not None else f"{series_name}/{dataset['path']}",
                    axis_labels=layer_axes,
                    scale=[scale[axes.index(axis)] for axis in layer_axes],
                    contrast_limits=limits if channel_axis is not None else limits[0],
                    rgb=False,
                )
        if not viewer.layers:
            raise ValueError("The archive contains no supported CZI OME-Zarr image series.")
        napari.run()
    finally:
        viewer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())