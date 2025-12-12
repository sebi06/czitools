from typing import Union, List, Optional
import os
from dataclasses import dataclass, field
from box import Box, BoxList
from czitools.utils import logging_tools
from czitools.utils.box import get_czimd_box

logger = logging_tools.set_logging()


@dataclass
class CziDetector:
    """Extract detector information from a CZI file.

    This class converts the detector information found under
    `ImageDocument.Metadata.Information.Instrument.Detectors.Detector`
    into parallel lists for each detector attribute. The implementation
    is defensive: if the Instrument/Detectors block is missing the
    instance will contain single-element lists with ``None`` so callers
    that expect list-shaped fields keep working.

    Attributes:
        czisource: Path/URL string or already-parsed `Box` with metadata.
        model, name, Id, modeltype, gain, zoom, amplificationgain:
            Lists containing values per-detector (may be a single None).
        verbose: If True, log informational messages while parsing.
    """

    czisource: Union[str, os.PathLike, Box]
    model: List[Optional[str]] = field(init=False, default_factory=list)
    name: List[Optional[str]] = field(init=False, default_factory=list)
    Id: List[Optional[str]] = field(init=False, default_factory=list)
    modeltype: List[Optional[str]] = field(init=False, default_factory=list)
    gain: List[Optional[float]] = field(init=False, default_factory=list)
    zoom: List[Optional[float]] = field(init=False, default_factory=list)
    amplificationgain: List[Optional[float]] = field(init=False, default_factory=list)
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.verbose:
            logger.info("Reading Detector information from CZI image metadata.")

        czi_box = self.czisource if isinstance(self.czisource, Box) else get_czimd_box(self.czisource)

        # Instrument block may be absent in some CZI files; handle gracefully
        instr = getattr(czi_box.ImageDocument.Metadata.Information, "Instrument", None)
        if instr is None:
            if self.verbose:
                logger.info("No Detector(s) information found in Instrument block.")
            # Keep historical single-None placeholders
            placeholder = [None]
            self.model = placeholder.copy()
            self.name = placeholder.copy()
            self.Id = placeholder.copy()
            self.modeltype = placeholder.copy()
            self.gain = placeholder.copy()
            self.zoom = placeholder.copy()
            self.amplificationgain = placeholder.copy()
            return

        # Get Detectors.Detector which can be a Box (single) or BoxList (multiple)
        detectors = getattr(instr, "Detectors", None)
        det = getattr(detectors, "Detector", None) if detectors is not None else None

        if det is None:
            if self.verbose:
                logger.info("Instrument present but no Detector entries found under Instrument.Detectors.")
            return

        def _append_from_box(b: Box) -> None:
            self.Id.append(getattr(b, "Id", None))
            self.name.append(getattr(b, "Name", None))
            self.model.append(getattr(b, "Model", None))
            self.modeltype.append(getattr(b, "Type", None))
            self.gain.append(getattr(b, "Gain", None))
            self.zoom.append(getattr(b, "Zoom", None))
            self.amplificationgain.append(getattr(b, "AmplificationGain", None))

        # Single detector
        if isinstance(det, Box):
            _append_from_box(det)

        # Multiple detectors
        elif isinstance(det, BoxList):
            for d in det:
                _append_from_box(d)
