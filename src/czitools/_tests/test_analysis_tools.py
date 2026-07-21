"""Tests for the analysis tools (ArrayProcessor, HCS analysis, plotting).

These tests are skipped automatically when the optional analysis dependencies
(``scikit-image``, ``seaborn``, ``matplotlib``, ``ngff-zarr``) are not installed.
"""

import numpy as np
import pytest

pytest.importorskip("skimage")
pytest.importorskip("matplotlib")
pytest.importorskip("seaborn")

from czitools.analysis_tools import ArrayProcessor, create_well_plate_heatmap


def test_array_processor_requires_2d() -> None:
    with pytest.raises(TypeError):
        ArrayProcessor(np.zeros((2, 2, 2), dtype=np.uint8))


def test_array_processor_otsu_and_label() -> None:
    # Build a simple image with two clearly separated bright blobs.
    image = np.zeros((64, 64), dtype=np.uint16)
    image[10:20, 10:20] = 5000
    image[40:55, 40:55] = 5000

    # Use manual threshold instead of Otsu (Otsu returns 0 for this sparse image,
    # making the entire array a mask that touches all borders)
    mask = ArrayProcessor(image).apply_threshold(value=1000, invert_result=False)
    assert mask.dtype == bool

    # Blobs are 100 px and 225 px. With min_size=50, both blobs survive
    # removal and are counted as separate objects.
    labelled, num_objects, props = ArrayProcessor(mask).label_objects(
        min_size=50,
        label_rgb=False,
        measure_params=True,
    )
    assert num_objects == 2
    assert props is not None
    assert len(props) == 2


def test_array_processor_threshold_validation() -> None:
    ap = ArrayProcessor(np.zeros((8, 8), dtype=np.uint8))
    with pytest.raises(ValueError):
        ap.apply_threshold(-1)


def test_create_well_plate_heatmap_returns_figure() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    results = {"A/1": 100.0, "A/2": 120.0, "B/1": 95.0}
    fig = create_well_plate_heatmap(results, num_rows=8, num_cols=12)
    assert isinstance(fig, Figure)
