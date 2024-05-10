import pandas as pd
import unittest
import os
from czitools.tools import misc


class TestSavePlanetTable(unittest.TestCase):

    def test_save_planetable(self):
        # Create a test dataframe
        test_df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        # Define the test filename
        test_filename = 'test.csv'
        # Call the save_planetable function
        result = misc.save_planetable(
            df=test_df, filename=test_filename, separator=';', index=False)
        # Check that the result matches the expected filename
        self.assertEqual(result, 'test_planetable.csv')
        # Check that the file was created
        self.assertTrue(os.path.exists('test_planetable.csv'))
        # Load the CSV file and check that the data matches the original dataframe
        loaded_df = pd.read_csv('test_planetable.csv', sep=';')
        self.assertTrue(loaded_df.equals(test_df))

        # Clean up the created file
        os.remove('test_planetable.csv')
