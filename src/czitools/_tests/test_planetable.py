import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from czitools.utils import planetable

basedir = Path(__file__).resolve().parents[3]


@pytest.mark.usefixtures("planetable_dict")
class TestPlaneTable(unittest.TestCase):
    def test_save_planetable(self):

        # Create a test dataframe
        test_df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        # Define the test filename
        test_filename = "test.csv"
        # Call the save_planetable function
        result = planetable.save_planetable(df=test_df, filepath=test_filename, separator=";", index=False)
        # Check that the result matches the expected filename
        self.assertEqual(result, "test_planetable.csv")
        # Check that the file was created
        self.assertTrue(os.path.exists("test_planetable.csv"))
        # Load the CSV file and check that the data matches the original dataframe
        loaded_df = pd.read_csv("test_planetable.csv", sep=";")
        self.assertTrue(loaded_df.equals(test_df))

        # Clean up the created file
        os.remove("test_planetable.csv")

    def test_read_planetable_from_fixture(self):
        """Use the predefined planetable dict (attached as self.planetable_dict by conftest.py)
        to build a DataFrame and compare it to the DataFrame returned by get_planetable.
        """
        # Build DataFrame from the fixture dict provided by conftest.py
        df_fixture = pd.DataFrame(self.planetable_dict)

        # Call get_planetable for the same CZI used in the CSV-based test
        czifile = "CellDivision_T3_Z5_CH2_X240_Y170.czi"
        filepath_czi = basedir / "data" / czifile
        czi_df, _ = planetable.get_planetable(
            filepath_czi, save_table=False, norm_time=True, planes={"time": 0, "channel": 0}
        )

        # Compare the DataFrames
        self.assertTrue(df_fixture.equals(czi_df))


@pytest.fixture
def pyramid_subblocks(monkeypatch: pytest.MonkeyPatch) -> list[SimpleNamespace]:
    """Provide interleaved resolutions without reading or decoding image pixels."""
    subblocks = []
    for scene, channel in ((0, 0), (1, 1)):
        for downsample, xstart in ((1, 0), (2, 0), (1, 100), (2, 100), (4, 0), (4, 100)):
            entry = SimpleNamespace(
                dims=("C", "Y", "X"),
                start=(channel, 20, xstart),
                shape=(1, 80, 100),
                stored_shape=(1, 80 // downsample, 100 // downsample),
                scene_index=scene,
                is_pyramid=downsample != 1,
            )
            subblocks.append(SimpleNamespace(directory_entry=entry, metadata=lambda: "<METADATA/>"))
    reader = MagicMock()
    reader.__enter__.return_value.subblocks.side_effect = lambda: iter(subblocks)
    monkeypatch.setattr(planetable.czifile_module, "CziFile", lambda _: reader)
    monkeypatch.setattr(planetable, "HAS_PROGRESSBAR", False)
    return subblocks


def test_include_pyramid_preserves_full_resolution(pyramid_subblocks: list[SimpleNamespace]) -> None:
    """Include every resolution without altering full-resolution tile indices or bounds."""
    default, _ = planetable.get_planetable("image.czi", norm_time=False)
    explicit_default, _ = planetable.get_planetable("image.czi", norm_time=False, include_pyramid=False)
    result, _ = planetable.get_planetable("image.czi", norm_time=False, include_pyramid=True)

    pd.testing.assert_frame_equal(default, explicit_default)
    assert len(result) == len(pyramid_subblocks)
    assert "is_pyramid" not in default.columns
    assert result.columns.tolist() == [
        *default.columns,
        "is_pyramid",
        "stored_width",
        "stored_height",
        "downsample_x",
        "downsample_y",
    ]
    full_resolution = result.loc[~result["is_pyramid"], default.columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(full_resolution.drop(columns="Subblock"), default.drop(columns="Subblock"))
    assert result["Subblock"].tolist() == list(range(len(result)))
    assert result["width"].eq(100).all()
    assert result["height"].eq(80).all()
    assert result["ystart"].eq(20).all()
    assert set(result["xstart"]) == {0, 100}
    assert result["downsample_x"].equals(result["downsample_y"])
    assert set(result["downsample_x"]) == {1.0, 2.0, 4.0}
    assert (result["stored_width"] * result["downsample_x"]).eq(result["width"]).all()
    assert (result["stored_height"] * result["downsample_y"]).eq(result["height"]).all()
    for _, resolution in result.groupby(["S", "T", "C", "Z", "downsample_x", "downsample_y"]):
        assert resolution["M"].tolist() == [0, 1]


@pytest.mark.parametrize(
    ("planes", "expected_rows"),
    [({"tile": 1}, 6), ({"scene": 1, "channel": 1}, 6), ({"time": 9}, 0)],
)
def test_include_pyramid_filter_and_save(
    pyramid_subblocks: list[SimpleNamespace],
    tmp_path: Path,
    planes: dict[str, int],
    expected_rows: int,
) -> None:
    """Keep pyramid metadata and sequential row numbers through filtering and CSV export."""
    result, csv_path = planetable.get_planetable(
        tmp_path / "image.czi",
        planes=planes,
        include_pyramid=True,
        save_table=True,
        table_index=False,
    )

    assert len(result) == expected_rows
    assert result["Subblock"].tolist() == list(range(expected_rows))
    assert str(result["is_pyramid"].dtype) == "bool"
    assert str(result["stored_width"].dtype) == "int32"
    assert str(result["stored_height"].dtype) == "int32"
    assert str(result["downsample_x"].dtype) == "float64"
    assert str(result["downsample_y"].dtype) == "float64"
    assert csv_path is not None
    if expected_rows:
        pd.testing.assert_frame_equal(pd.read_csv(csv_path, sep=";"), result, check_dtype=False)
    else:
        assert pd.read_csv(csv_path, sep=";").columns.tolist() == result.columns.tolist()


@pytest.mark.parametrize("subblock_selection", ["empty", "full_resolution", "pyramid_only"])
def test_include_pyramid_subblock_selection(pyramid_subblocks: list[SimpleNamespace], subblock_selection: str) -> None:
    """Handle empty inputs and files containing only one subblock category."""
    if subblock_selection == "empty":
        pyramid_subblocks.clear()
    else:
        pyramid_subblocks[:] = [
            subblock
            for subblock in pyramid_subblocks
            if subblock.directory_entry.is_pyramid == (subblock_selection == "pyramid_only")
        ]
    result, _ = planetable.get_planetable("image.czi", include_pyramid=True)

    assert len(result) == len(pyramid_subblocks)
    assert str(result["is_pyramid"].dtype) == "bool"
    if subblock_selection == "full_resolution":
        assert not result["is_pyramid"].any()
        assert result["downsample_x"].eq(1.0).all()
        assert result["downsample_y"].eq(1.0).all()
    elif subblock_selection == "pyramid_only":
        assert result["is_pyramid"].all()


def test_include_pyramid_anisotropic_downsampling(pyramid_subblocks: list[SimpleNamespace]) -> None:
    """Report actual X/Y ratios without assuming equal or power-of-two downsampling."""
    entry = pyramid_subblocks[1].directory_entry
    entry.shape = (1, 81, 101)
    entry.stored_shape = (1, 27, 50)
    result, _ = planetable.get_planetable("image.czi", include_pyramid=True)
    row = result.loc[(result["S"] == 0) & (result["stored_height"] == 27)].iloc[0]

    assert row["width"] == 101
    assert row["height"] == 81
    assert row["downsample_x"] == pytest.approx(101 / 50)
    assert row["downsample_y"] == pytest.approx(3.0)


@pytest.mark.parametrize("stored_width", [0, -1])
def test_include_pyramid_invalid_dimensions(pyramid_subblocks: list[SimpleNamespace], stored_width: int) -> None:
    """Reject non-positive stored sizes instead of dividing by zero or returning invalid scales."""
    pyramid_subblocks[1].directory_entry.stored_shape = (1, 40, stored_width)
    with pytest.raises(ValueError, match="Subblock 1 has non-positive spatial dimensions"):
        planetable.get_planetable("image.czi", include_pyramid=True)
