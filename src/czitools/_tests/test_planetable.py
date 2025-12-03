import pandas as pd
import unittest
import os
from czitools.utils import planetable
from pathlib import Path
import pytest

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
        czi_df, _ = planetable.get_planetable(filepath_czi, save_table=False, norm_time=True, planes={"time":0, "channel":0})

        # Compare the DataFrames
        self.assertTrue(df_fixture.equals(czi_df))

