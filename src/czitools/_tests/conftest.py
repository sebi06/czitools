import pytest
import pandas as pd


@pytest.fixture
def df():
    dataframe = pd.DataFrame({"Time [s]": [1, 2, 3, 4], "Value": [10, 20, 30, 40]})
    return dataframe


@pytest.fixture
def planetable():

    pt = pd.DataFrame(
        {
            "S": [0, 0, 1, 1, 2],
            "T": [0, 1, 2, 3, 4],
            "C": [0, 1, 3, 5, 7],
            "Z": [0, 1, 2, 3, 5],
        }
    )
    return pt
