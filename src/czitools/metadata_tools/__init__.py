"""Tools for extracting and working with CZI metadata.

Provides dataclasses for reading all major metadata sections from a CZI file,
including dimensions, scaling, channel info, bounding boxes, objectives,
detectors, microscope info, sample info, and attachment metadata.

Typical usage:

```python
from czitools.metadata_tools import CziMetadata

mdata = CziMetadata("my_image.czi")
```
"""

from .add_metadata import CziAddMetaData
from .attachment import CziAttachments
from .boundingbox import CziBoundingBox
from .channel import CziChannelInfo, hex_to_rgb
from .czi_metadata import (
    CziMetadata,
    create_md_dict_nested,
    create_md_dict_red,
    get_metadata_as_object,
    writexml,
)
from .detector import CziDetector
from .dimension import CziDimensions
from .helper import AttachmentType, DictObj, ValueRange
from .microscope import CziMicroscope
from .objective import CziObjectives
from .sample import CziSampleInfo, get_scenes_for_well
from .scaling import CziScaling
from .scene import CziScene

__all__ = [
    "CziAddMetaData",
    "CziAttachments",
    "CziBoundingBox",
    "CziChannelInfo",
    "hex_to_rgb",
    "CziMetadata",
    "get_metadata_as_object",
    "writexml",
    "create_md_dict_nested",
    "create_md_dict_red",
    "CziDetector",
    "CziDimensions",
    "ValueRange",
    "AttachmentType",
    "DictObj",
    "CziMicroscope",
    "CziObjectives",
    "CziSampleInfo",
    "get_scenes_for_well",
    "CziScaling",
    "CziScene",
]
