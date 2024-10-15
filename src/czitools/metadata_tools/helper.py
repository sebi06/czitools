from dataclasses import dataclass, field
from enum import Enum


@dataclass
class ValueRange:
    lo: float
    hi: float


class AttachmentType(Enum):
    SlidePreview = 1
    Label = 2
    Prescan = 3


class DictObj:
    """
    This class recursively converts a dictionary into an object, allowing attribute-style access to dictionary keys.
    See https://joelmccune.com/python-dictionary-as-object/
    Attributes:
        in_dict (dict): The dictionary to be converted into an object.
    Methods:
        __init__(in_dict: dict) -> None:
            Initializes the DictObj instance by converting the input dictionary into an object.
    """

    # TODO: is this class still needed because of using python-box

    def __init__(self, in_dict: dict) -> None:
        assert isinstance(in_dict, dict)

        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
                setattr(
                    self, key, [DictObj(x) if isinstance(x, dict) else x for x in val]
                )
            else:
                setattr(self, key, DictObj(val) if isinstance(val, dict) else val)
