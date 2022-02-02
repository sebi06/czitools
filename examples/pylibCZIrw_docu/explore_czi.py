from _pylibCZIrw import czi_reader
from _pylibCZIrw import DimensionIndex
from _pyliBCZIrw import

czifile = r"testdata/w96_A1+A2.czi"

czi_img = czi_reader(czifile)

czi_img.GetChannelPixelType(0)

dim = DimensionIndex.S  # Here the variable dim will represent the S (scene) Dimension
print(czi_img.GetDimensionSize(dim))

stats = czi_img.GetSubBlockStats()

print("Done.")
