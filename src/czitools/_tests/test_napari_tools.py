import pytest

try:
    import napari
    from czitools.napari_tools import napari_tools
    from czitools.metadata_tools import czi_metadata

    NAPARI_INSTALLED = True
except (ImportError, ModuleNotFoundError) as error:
    NAPARI_INSTALLED = False


from czitools.read_tools import read_tools
import numpy as np
from czitools.metadata_tools.czi_metadata import CziMetadata
from pathlib import Path
from typing import Dict, Tuple, Literal
import os

# check if the test in executed as part of a GITHUB action
IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

basedir = Path(__file__).resolve().parents[3]


@pytest.mark.skipif(not NAPARI_INSTALLED, reason="Napari is not installed.")
@pytest.mark.parametrize(
    "dim_order, sliders, expected_sliders",
    [
        ({"S": 0, "T": 1, "Z": 2}, ("S", "T", "Z"), ("S", "T", "Z")),
    ],
)
def test_rename_sliders(
    dim_order: Dict[str, int], sliders: Tuple[str], expected_sliders: Tuple[str]
) -> None:
    """Test that "rename_sliders" correctly renames the sliders based on the `dim_order` dict."""

    # Act
    renamed_sliders = napari_tools.rename_sliders(sliders, dim_order)

    # Assert
    assert renamed_sliders == expected_sliders


# exclude the test when executed inside a GITHUB action
@pytest.mark.skipif(not NAPARI_INSTALLED, reason="Napari is not installed.")
@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
@pytest.mark.parametrize(
    "czifile, num_layers, show_metadata, wdname",
    [
        ("w96_A1+A2.czi", 2, "tree", "MetadataTree"),
        ("CellDivision_T3_Z5_CH2_X240_Y170.czi", 2, "table", "MetadataTable"),
    ],
)
def test_show_image(
    czifile: str,
    num_layers: int,
    show_metadata: Literal["none", "tree", "table"],
    wdname: str,
) -> None:
    """Test that the `show` function correctly displays a two-channel image and the metadada widgets."""

    filepath = basedir / "data" / czifile
    #md = CziMetadata(filepath)

    array6d, mdata = read_tools.read_6darray(
        filepath,
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
        blending="additive",
        contrast="from_czi",
        gamma=0.85,
        show_metadata=show_metadata,
        name_sliders=True,
    )

    # Check that a tree widget is visible in the viewer
    assert (wdname in viewer.window.__dict__["_dock_widgets"].data.keys()) is True

    # Check that the layer is present in the viewer
    assert len(viewer.layers) == num_layers

    for layer in range(num_layers):
        # Check that the layer is a `napari.layers.Image` layer
        assert isinstance(viewer.layers[layer], napari.layers.Image)

        # Check that the layer's data is the same as the input image
        np.testing.assert_array_equal(
            viewer.layers[layer].data, array6d[:, :, layer : layer + 1, :, :, :]
        )

    viewer.close()
