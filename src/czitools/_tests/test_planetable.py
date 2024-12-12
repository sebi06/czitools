import pandas as pd
import unittest
import os
from czitools.utils import misc
from pathlib import Path

basedir = Path(__file__).resolve().parents[3]


class TestPlaneTable(unittest.TestCase):

    def test_save_planetable(self):

        # Create a test dataframe
        test_df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        # Define the test filename
        test_filename = "test.csv"
        # Call the save_planetable function
        result = misc.save_planetable(
            df=test_df, filename=test_filename, separator=";", index=False
        )
        # Check that the result matches the expected filename
        self.assertEqual(result, "test_planetable.csv")
        # Check that the file was created
        self.assertTrue(os.path.exists("test_planetable.csv"))
        # Load the CSV file and check that the data matches the original dataframe
        loaded_df = pd.read_csv("test_planetable.csv", sep=";")
        self.assertTrue(loaded_df.equals(test_df))

        # Clean up the created file
        os.remove("test_planetable.csv")

    def test_read_planetable(self):

        cvs = [
            "S2_3x3_CH2_planetable.csv",
            "CellDivision_T3_Z5_CH2_X240_Y170_planetable.csv",
        ]
        czi = ["S2_3x3_CH2.czi", "CellDivision_T3_Z5_CH2_X240_Y170.czi"]

        for csvfile, czifile in zip(cvs, czi):

            # get the CZI filepath
            filepath_table = basedir / "data" / csvfile
            filepath_czi = basedir / "data" / czifile

            # Load the CSV file and check that the data matches the original dataframe
            loaded_df = pd.read_csv(filepath_table, sep=";", index_col=0)
            czi_df = misc.get_planetable(filepath_czi, norm_time=True, pt_complete=True)

            # differences = loaded_df.compare(czi_df)

            self.assertTrue(loaded_df.equals(czi_df))


# test_run = TestPlaneTable()
# test_run.test_read_planetable()
