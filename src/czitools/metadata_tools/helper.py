from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict


@dataclass
class ValueRange:
    """Numeric range container with low and high floats."""

    lo: float
    hi: float


class AttachmentType(Enum):
    """Enumeration for known attachment types in CZI files."""

    SlidePreview = 1
    Label = 2
    Prescan = 3


class DictObj:
    """Lightweight helper to access nested dictionaries with attribute syntax.

    This recursively converts dictionaries into objects so callers can use
    ``obj.key`` instead of ``dict['key']``. Lists and tuples are preserved,
    but dictionary elements inside them are converted as well.

    Note: the project uses `python-box` in many places; this helper exists
    primarily for simple conversions/tests and may be removed in favor of
    `Box` if no special behavior is required.

    Args:
        in_dict: Input mapping to convert. Must be a dict.
    """

    # TODO: consider deprecating this helper in favor of `box.Box`

    def __init__(self, in_dict: Dict[str, Any]) -> None:
        if not isinstance(in_dict, dict):
            raise AssertionError("DictObj expects a dictionary as input")

        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
                # Convert dict elements inside lists/tuples, preserve other types
                setattr(self, key, [DictObj(x) if isinstance(x, dict) else x for x in val])
            else:
                setattr(self, key, DictObj(val) if isinstance(val, dict) else val)
