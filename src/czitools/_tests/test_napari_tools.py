import pytest
from czitools import napari_tools
from typing import List, Dict, Tuple, Optional, Type, Any, Union, Mapping


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
