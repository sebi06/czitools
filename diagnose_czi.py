# -*- coding: utf-8 -*-
"""
Diagnostic script to investigate the CZI file that's causing issues
"""

from pathlib import Path
from czitools.metadata_tools.czi_metadata import CziMetadata
from czitools.metadata_tools.dimension import CziDimensions
from czitools.metadata_tools.scaling import CziScaling
import pylibCZIrw.czi as pyczi
import xmltodict

filepath = r"F:\Testdata_Zeiss\bug_testing\nikos\PSF Pln Apo 63x1.4 Oil 488nm.czi"

print(f"Analyzing file: {filepath}")
print("=" * 80)

# Get metadata using czitools
try:
    # Use specific metadata classes
    dims = CziDimensions(filepath)
    scaling = CziScaling(filepath)

    print("\n✅ CziMetadata extraction successful")
    print(f"Time: SizeT={dims.SizeT}")
    print(f"Channels: SizeC={dims.SizeC}")
    print(f"Z-Slices: SizeZ={dims.SizeZ}")
    print(f"Image size: SizeY={dims.SizeY}, SizeX={dims.SizeX}")

    # Get raw XML metadata
    print("\nExtracting XML metadata...")
    czidoc_meta = pyczi.CziReader(filepath)
    metadata_dict = czidoc_meta.metadata
    czidoc_meta.close()

    # Check for compression
    try:
        image_info = metadata_dict["ImageDocument"]["Metadata"]["Information"]["Image"]
        if "Compression" in image_info:
            print(f"Compression: {image_info['Compression']}")
        else:
            print("Compression: None or not specified")

        # Check for pixel type
        if "ComponentBitCount" in image_info:
            print(f"ComponentBitCount: {image_info['ComponentBitCount']}")
        if "PixelType" in image_info:
            print(f"PixelType: {image_info['PixelType']}")

        # Print entire image info
        print(f"\nFull Image Info:")
        for key, value in image_info.items():
            print(f"  {key}: {value}")

    except KeyError as ke:
        print(f"Could not extract image info: {ke}")
        print(f"Available keys: {metadata_dict.keys()}")

except Exception as e:
    print(f"❌ Metadata extraction failed: {e}")
    import traceback

    traceback.print_exc()

# Try to open with pylibCZIrw directly
print("\n" + "=" * 80)
print("Testing pylibCZIrw direct access:")
try:
    czidoc = pyczi.CziReader(filepath)
    print("✅ CziReader opened successfully")

    # Try to read the first plane
    print("\nAttempting to read first plane (T=0, Z=0, C=0)...")
    try:
        image2d = czidoc.read(plane={"T": 0, "Z": 0, "C": 0})
        print(f"✅ First plane read successfully, shape: {image2d.shape}")
    except Exception as e:
        print(f"❌ Failed to read first plane: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback

        traceback.print_exc()

    # Try with zoom parameter
    print("\nAttempting to read first plane with zoom=1.0...")
    try:
        image2d = czidoc.read(plane={"T": 0, "Z": 0, "C": 0}, zoom=1.0)
        print(f"✅ First plane read with zoom successfully, shape: {image2d.shape}")
    except Exception as e:
        print(f"❌ Failed to read first plane with zoom: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback

        traceback.print_exc()

    # Try reading without zoom or scaling
    print("\nAttempting to read mosaic frame...")
    try:
        image2d = czidoc.read_mosaic_frame(C=0, T=0, Z=0)
        print(f"✅ read_mosaic_frame successful, shape: {image2d.shape}")
    except Exception as e:
        print(f"❌ read_mosaic_frame failed: {e}")

    czidoc.close()
    print("\n✅ CziReader closed")

except Exception as e:
    print(f"❌ pylibCZIrw access failed: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 80)
print("Checking for special CZI characteristics...")

# Check if it's using aicspylibczi
try:
    from aicspylibczi import CziFile

    print("\nTrying with aicspylibczi:")
    czi = CziFile(filepath)
    print(f"✅ aicspylibczi opened successfully")
    print(f"Dims: {czi.dims}")
    print(f"Size: {czi.size}")
    print(f"Is mosaic: {czi.is_mosaic()}")

    # Try to read with aicspylibczi
    try:
        data = czi.read_image()
        print(f"✅ aicspylibczi read successful, shape: {data[0].shape}")
    except Exception as e:
        print(f"⚠️ aicspylibczi read failed: {e}")

except ImportError:
    print("aicspylibczi not available")
except Exception as e:
    print(f"❌ aicspylibczi failed: {e}")

print("\n" + "=" * 80)
print("Diagnosis complete")
