from pylibCZIrw import czi as pyczi
import sys
import numpy as np
from tqdm.contrib.itertools import product

# show the used python env
print("Using:", sys.executable)

filename = r"data/test_write_metadata.czi"

# create a new CZI file
channel_names = {0: "Ch1", 1: "Ch2", 2: "Ch3"}
cmp = "zstd1:Preprocess=HiLoByteUnpack;Level=Best"
scale_xy = 0.1  # micron per pixel
scale_z = 0.3  # micron per plane
size_s = 2
size_t = 4
size_z = 5
size_c = 3
size_x = 200
size_y = 100
total_frames = size_s * size_t * size_c * size_z

# create mutil-dimensional array
img = np.random.randint(
    0, 256, size=(size_s, size_t, size_c, size_z, size_y, size_x), dtype=np.uint8
)

# open new CZI file for writing
with pyczi.create_czi(filename, exist_ok=True, compression_options=cmp) as czidoc_w:

    # iterate over all 2D planes
    for s, t, c, z in product(
        range(size_s),
        range(size_t),
        range(size_c),
        range(size_z),
        desc="Reading 2D planes",
        unit=" 2Dplanes",
    ):

        # write a 2D plane to the CZI file
        czidoc_w.write(
            data=img[s, t, c, z, :, :],
            plane={"C": c, "T": t, "Z": z},
            scene=s,
            location=(512 * s, 512 * s),
        )

    # write the document title, channel names, custom attributes and XYZ scaling to the CZI file
    czidoc_w.write_metadata(
        document_name=filename,
        channel_names=channel_names,
        custom_attributes={"key1": 0, "key2": 1},
        display_settings={
            0: pyczi.ChannelDisplaySettingsDataClass(
                True,
                pyczi.TintingMode.Color,
                pyczi.Rgb8Color(np.uint8(255), np.uint8(0), np.uint8(0)),
                0.2,
                0.8,
            ),
            1: pyczi.ChannelDisplaySettingsDataClass(
                True,
                pyczi.TintingMode.Color,
                pyczi.Rgb8Color(np.uint8(), np.uint8(255), np.uint8(0)),
                0.2,
                0.8,
            ),
            2: pyczi.ChannelDisplaySettingsDataClass(
                True,
                pyczi.TintingMode.Color,
                pyczi.Rgb8Color(np.uint8(0), np.uint8(0), np.uint8(255)),
                0.2,
                0.8,
            ),
        },
        scale_x=scale_xy * 10**-6,
        scale_y=scale_xy * 10**-6,
        scale_z=scale_z * 10**-6,
    )


# read the CZI file again
with pyczi.open_czi(filename) as czidoc:

    md_dict = czidoc.metadata

    custom_attributes = md_dict["ImageDocument"]["Metadata"]["Information"][
        "CustomAttributes"
    ]
    print(f"Custom Attributes: {custom_attributes}")
