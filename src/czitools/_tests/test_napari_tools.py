import pytest
from czitools import napari_tools, metadata_tools, read_tools
import numpy as np
import napari
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Type, Any, Union, Mapping

basedir = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "dim_order, sliders, expected_sliders",
    [
        ({"S": 0, "T": 1, "Z": 2}, ("S", "T", "Z"), ("S", "T", "Z")),
    ],
)
def test_rename_sliders(
    dim_order: Dict[str, int], sliders: Tuple[str], expected_sliders: Tuple[str]
) -> None:
    """Test that `rename_sliders` correctly renames the sliders based on the `dim_order` dict."""

    # Act
    renamed_sliders = napari_tools.rename_sliders(sliders, dim_order)

    # Assert
    assert renamed_sliders == expected_sliders


@pytest.mark.parametrize(
    "czifile, num_layers",
    [
        ("w96_A1+A2.czi", 2),
    ],
)
def test_show_image(czifile: str, num_layers: int) -> None:
    """Test that the `show` function correctly displays a two-channel image."""

    filepath = basedir / "data" / czifile
    md = metadata_tools.CziMetadata(filepath)

    array6d, mdata, dim_string6d = read_tools.read_6darray(
        filepath,
        output_order="STCZYX",
        use_dask=False,
        chunk_zyx=False,
    )

    # Create a viewer
    viewer = napari.Viewer()

    # Display the image
    layers = napari_tools.show(
        viewer,
        array6d,
        mdata,
        dim_string=dim_string6d,
        blending="additive",
        contrast="from_czi",
        gamma=0.85,
        show_metadata="tree",
        name_sliders=True,
    )

    # Check that the layer is present in the viewer
    assert len(viewer.layers) == num_layers

    for layer in range(num_layers):
        # Check that the layer is a `napari.layers.Image` layer
        assert isinstance(viewer.layers[layer], napari.layers.Image)

        # Check that the layer's data is the same as the input image
        np.testing.assert_array_equal(viewer.layers[layer].data, array6d[:, :, layer:layer+1, :, :, :])

    viewer.close()
