# -*- coding: utf-8 -*-

from pylibCZIrw import czi as pyczi

filepath = r"F:\Testdata_Zeiss\CZI_Testfiles\DAPI_GFP.czi"


class ZenReader:

    def __init__(self, filepath):
        self.filepath = filepath
        self._czidoc = pyczi.CziReader(self.filepath)

    def read_czi(self):
        total_bounding_box = self._czidoc.total_bounding_box
        print(total_bounding_box)

    def __del__(self):
        self._czidoc.close()
