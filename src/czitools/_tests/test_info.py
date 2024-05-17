from czitools.metadata_tools import czi_metadata as czimd
from pathlib import Path
import pytest
from typing import List

basedir = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "czifile, result",
    [
        (
            "CellDivision_T3_Z5_CH2_X240_Y170.czi",
            [
                "ZEN 3.2 (blue edition)",
                "3.2.0.00001",
                "2016-02-12T09:41:02.4915604Z",
                True,
                "CellDivision_T3_Z5_CH2_X240_Y170.czi",
            ],
        ),
        (
            "Airyscan.czi",
            ["ZEN (blue edition)", "3.5.093.00000", None, True, "Airyscan.czi"],
        ),
        ("newCZI_zloc.czi", ["pylibCZIrw", "3.2.1", None, True, "newCZI_zloc.czi"]),
    ],
)
def test_information(czifile: str, result: List) -> None:
    filepath = basedir / "data" / czifile
    md = czimd.CziMetadata(filepath)

    assert md.software_name == result[0]
    assert md.software_version == result[1]
    assert md.acquisition_date == result[2]
    assert Path.exists(Path(md.dirname)) == result[3]
    assert md.filename == result[4]
