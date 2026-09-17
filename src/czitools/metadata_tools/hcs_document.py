"""Versioned, JSON-safe metadata document for CZI HCS acquisitions."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Mapping

from pydantic import TypeAdapter, ValidationError

from czitools import __version__
from czitools.metadata_tools.hcs import CziPlate, normalize_well_name

if TYPE_CHECKING:
    from czitools.metadata_tools.czi_metadata import CziMetadata

CZI_HCS_DOCUMENT_SCHEMA_VERSION = "1.0"

_AXES = (
    ("S", "Scene"),
    ("T", "Time"),
    ("C", "Channel"),
    ("Z", "Z-slice"),
    ("Y", "Height"),
    ("X", "Width"),
    ("M", "Mosaic"),
    ("R", "Rotation"),
    ("I", "Illumination"),
    ("H", "Phase"),
    ("V", "View"),
    ("B", "Block"),
)


@dataclass(frozen=True)
class SourceInfo:
    """Stable identity of the source CZI."""

    uri: str
    filename: str
    suffix: str
    source_kind: str
    size_bytes: int | None
    sha256: str | None


@dataclass(frozen=True)
class ProvenanceInfo:
    """Extractor identity and generation time (volatile provenance)."""

    extractor: str
    extractor_version: str
    generated_at_utc: str


@dataclass(frozen=True)
class AcquisitionInfo:
    """File-level acquisition properties."""

    acquisition_date: str | None
    creation_date: str | None
    software_name: str | None
    software_version: str | None
    user_name: str | None
    has_scenes: bool
    is_mosaic: bool
    pixel_types: dict[str, str]
    consistent_pixel_types: bool | None


@dataclass(frozen=True)
class AxisInfo:
    """One physically observed image axis."""

    code: str
    meaning: str
    size: int | None
    start: int | None
    end: int | None
    valid_indices: tuple[int, ...]
    unit: str | None = None
    coordinates: tuple[float, ...] = ()


@dataclass(frozen=True)
class ImageInfo:
    """Stored image dimensions and bounds."""

    dimension_order: str
    axes: tuple[AxisInfo, ...]
    first_scene_shape_yx: tuple[int, int] | None
    total_bounding_box: dict[str, tuple[int, int]]
    scene_shape_is_consistent: bool
    scene_shape_tolerance_px: int


@dataclass(frozen=True)
class ChannelInfo:
    """Normalized metadata for one image channel."""

    index: int
    name: str | None
    dye: str | None
    dye_short: str | None
    description: str | None
    color: str | None
    black_point: float | None
    white_point: float | None
    gamma: float | None
    pixel_type: str | None
    is_rgb: bool | None


@dataclass(frozen=True)
class ScalingInfo:
    """Physical pixel scaling in named axes."""

    x: float | None
    y: float | None
    z: float | None
    unit: str | None
    value_source: str


@dataclass(frozen=True)
class MicroscopeInfo:
    """Microscope identity."""

    id: str | None
    name: str | None
    system: str | None


@dataclass(frozen=True)
class DetectorInfo:
    """Metadata for one detector."""

    index: int
    id: str | None
    name: str | None
    model: str | None
    model_type: str | None
    gain: float | None
    zoom: float | None
    amplification_gain: float | None


@dataclass(frozen=True)
class ObjectiveInfo:
    """Metadata for one objective."""

    index: int
    id: str | None
    name: str | None
    model: str | None
    immersion: str | None
    numerical_aperture: float | None
    nominal_magnification: float | None
    tube_lens_magnification: float | None
    total_magnification: float | None


@dataclass(frozen=True)
class InstrumentInfo:
    """Microscope, detector, objective, and scaling metadata."""

    microscope: MicroscopeInfo
    detectors: tuple[DetectorInfo, ...]
    objectives: tuple[ObjectiveInfo, ...]
    scaling: ScalingInfo


@dataclass(frozen=True)
class SampleInfo:
    """Sample carrier, specimen, and image-level stage position."""

    carrier: dict[str, Any] | None
    carrier_source: str | None
    specimen: dict[str, Any] | None
    specimen_source: str | None
    image_stage_x: float | None
    image_stage_y: float | None
    stage_unit: str


@dataclass(frozen=True)
class HcsInfo:
    """HCS detection result and declared/stored plate views."""

    detected: bool
    detection_reason: str
    declared_plate: CziPlate | None
    stored_plate: CziPlate | None
    stored_scene_indices: tuple[int, ...]
    declared_field_count: int
    stored_field_count: int


@dataclass(frozen=True)
class AttachmentInfo:
    """Summary of CZI attachments without binary payloads."""

    has_label: bool
    has_preview: bool
    has_prescan: bool
    names: tuple[str, ...]


@dataclass(frozen=True)
class QualityIssue:
    """One non-fatal metadata quality observation."""

    severity: str
    code: str
    message: str
    path: str | None = None


@dataclass(frozen=True)
class QualityInfo:
    """Overall metadata quality status and observations."""

    status: str
    issues: tuple[QualityIssue, ...]


@dataclass(frozen=True)
class CziHcsDocument:
    """One portable, versioned description of a CZI HCS acquisition."""

    schema_version: str
    source: SourceInfo
    provenance: ProvenanceInfo
    acquisition: AcquisitionInfo
    image: ImageInfo
    channels: tuple[ChannelInfo, ...]
    instrument: InstrumentInfo
    sample: SampleInfo
    hcs: HcsInfo
    attachments: AttachmentInfo
    quality: QualityInfo
    extensions: dict[str, Any]

    def __post_init__(self) -> None:
        """Validate structural invariants after construction."""
        if self.schema_version != CZI_HCS_DOCUMENT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported CZI HCS document schema version {self.schema_version!r}.")
        _validate_axes(self.image.axes)
        _validate_units(self.image, self.instrument.scaling)
        _validate_plate(self.hcs.declared_plate)
        _validate_plate(self.hcs.stored_plate)
        _validate_stored_fields(self.hcs)
        _validate_extensions(self.extensions)
        _validate_finite_values(asdict(self))

    @classmethod
    def from_metadata(
        cls,
        metadata: CziMetadata,
        *,
        include_checksum: bool = False,
        redact_user: bool = False,
        generated_at: datetime | None = None,
    ) -> CziHcsDocument:
        """Build a portable document from an extracted ``CziMetadata`` object.

        Args:
            metadata (CziMetadata): Extracted CZI metadata.
            include_checksum (bool): Calculate a SHA-256 hash for local files.
            redact_user (bool): Replace the acquisition user name with ``None``.
            generated_at (datetime | None): Fixed generation timestamp for
                reproducible output. Defaults to the current UTC time.

        Returns:
            CziHcsDocument: Validated machine-readable metadata document.
        """
        source = _build_source(metadata, include_checksum)
        provenance = _build_provenance(generated_at)
        acquisition = _build_acquisition(metadata, redact_user)
        image = _build_image(metadata)
        channels = _build_channels(metadata)
        instrument = _build_instrument(metadata)
        sample = _build_sample(metadata)
        hcs = _build_hcs(metadata)
        attachments = _build_attachments(metadata)
        quality = _build_quality(metadata, image, channels, hcs, instrument.scaling)
        return cls(
            schema_version=CZI_HCS_DOCUMENT_SCHEMA_VERSION,
            source=source,
            provenance=provenance,
            acquisition=acquisition,
            image=image,
            channels=channels,
            instrument=instrument,
            sample=sample,
            hcs=hcs,
            attachments=attachments,
            quality=quality,
            extensions={},
        )

    def to_dict(self, *, exclude_none: bool = False) -> dict[str, Any]:
        """Return a JSON-safe dictionary representation.

        Args:
            exclude_none (bool): Recursively omit dictionary keys whose value is
                ``None``. Defaults to False.

        Returns:
            dict[str, Any]: JSON-safe document payload.
        """
        payload = _to_json_value(asdict(self))
        if not isinstance(payload, dict):
            raise TypeError("The document root did not serialize to a dictionary.")
        return _drop_none(payload) if exclude_none else payload

    def to_json(self, *, indent: int | None = 2, exclude_none: bool = False) -> str:
        """Serialize the document to JSON with finite numbers only."""
        return json.dumps(
            self.to_dict(exclude_none=exclude_none),
            indent=indent,
            allow_nan=False,
            ensure_ascii=False,
        )

    def write_json(
        self,
        path: str | os.PathLike[str],
        *,
        indent: int | None = 2,
        exclude_none: bool = False,
    ) -> Path:
        """Write the document as UTF-8 JSON and return the output path."""
        output = Path(path)
        output.write_text(
            self.to_json(indent=indent, exclude_none=exclude_none) + "\n",
            encoding="utf-8",
        )
        return output

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CziHcsDocument:
        """Strictly validate and reconstruct a document from a mapping."""
        if not isinstance(payload, Mapping):
            raise TypeError("The document payload must be a mapping.")
        version = payload.get("schema_version")
        if version != CZI_HCS_DOCUMENT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported CZI HCS document schema version {version!r}; "
                f"expected {CZI_HCS_DOCUMENT_SCHEMA_VERSION!r}."
            )
        try:
            return _document_adapter().validate_python(dict(payload))
        except ValidationError as error:
            raise ValueError(f"Invalid CZI HCS document payload: {error}") from error

    @classmethod
    def read_json(cls, path: str | os.PathLike[str]) -> CziHcsDocument:
        """Read and validate a document from a UTF-8 JSON file."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("The JSON document root must be an object.")
        return cls.from_dict(payload)

    def iter_well_rows(self, *, stored: bool = True) -> Iterator[dict[str, Any]]:
        """Yield analytics rows with one row per well."""
        plate = self.hcs.stored_plate if stored else self.hcs.declared_plate
        if plate is None:
            return
        for well in plate.wells:
            yield {
                "plate_id": plate.id,
                "well_id": well.id,
                "well_name": well.canonical_name,
                "row_index": well.row_index,
                "column_index": well.column_index,
                "field_count": len(well.fields),
            }

    def iter_field_rows(self, *, stored: bool = True) -> Iterator[dict[str, Any]]:
        """Yield analytics rows with one row per field."""
        plate = self.hcs.stored_plate if stored else self.hcs.declared_plate
        if plate is None:
            return
        for well in plate.wells:
            for field in well.fields:
                yield {
                    "plate_id": plate.id,
                    "well_name": well.canonical_name,
                    **_to_json_value(asdict(field)),
                }

    def iter_channel_rows(self) -> Iterator[dict[str, Any]]:
        """Yield analytics rows with one row per channel."""
        for channel in self.channels:
            value = _to_json_value(asdict(channel))
            if isinstance(value, dict):
                yield value


def extract_hcs_document(
    filepath: str | os.PathLike[str],
    *,
    enrich_positions: bool = False,
    position_tolerance: float = 1.0,
    include_checksum: bool = False,
    redact_user: bool = False,
    generated_at: datetime | None = None,
) -> CziHcsDocument:
    """Extract a portable HCS metadata document from a CZI file.

    Args:
        filepath (str | os.PathLike[str]): Local CZI path or supported URL.
        enrich_positions (bool): Scan subblocks for stage and focus positions.
        position_tolerance (float): Position range above which a field is
            marked as conflicting.
        include_checksum (bool): Calculate SHA-256 for a local CZI.
        redact_user (bool): Omit the acquisition user name.
        generated_at (datetime | None): Fixed generation timestamp for
            reproducible output. Defaults to the current UTC time.

    Returns:
        CziHcsDocument: Validated machine-readable metadata document.
    """
    from czitools.metadata_tools.czi_metadata import CziMetadata

    metadata = CziMetadata(filepath, filter_hcs_to_stored_scenes=True)
    if enrich_positions:
        metadata.enrich_hcs_positions(position_tolerance)
    return CziHcsDocument.from_metadata(
        metadata,
        include_checksum=include_checksum,
        redact_user=redact_user,
        generated_at=generated_at,
    )


def _build_source(metadata: CziMetadata, include_checksum: bool) -> SourceInfo:
    filepath = str(metadata.filepath)
    is_url = bool(metadata.is_url)
    path = None if is_url else Path(filepath).resolve()
    return SourceInfo(
        uri=filepath if is_url else path.as_uri(),
        filename=metadata.filename or Path(filepath).name,
        suffix=Path(metadata.filename or filepath).suffix,
        source_kind="url" if is_url else "file",
        size_bytes=path.stat().st_size if path is not None and path.exists() else None,
        sha256=_sha256(path) if include_checksum and path is not None else None,
    )


def _build_provenance(generated_at: datetime | None) -> ProvenanceInfo:
    timestamp = generated_at or datetime.now(timezone.utc)
    return ProvenanceInfo(
        extractor="czitools",
        extractor_version=__version__,
        generated_at_utc=timestamp.astimezone(timezone.utc).isoformat(),
    )


def _build_acquisition(metadata: CziMetadata, redact_user: bool) -> AcquisitionInfo:
    return AcquisitionInfo(
        acquisition_date=metadata.acquisition_date,
        creation_date=metadata.creation_date,
        software_name=metadata.software_name,
        software_version=metadata.software_version,
        user_name=None if redact_user else metadata.user_name,
        has_scenes=bool(metadata.has_scenes),
        is_mosaic=bool(metadata.ismosaic),
        pixel_types={str(key): str(value) for key, value in (metadata.pixeltypes or {}).items()},
        consistent_pixel_types=metadata.consistent_pixeltypes,
    )


def _build_image(metadata: CziMetadata) -> ImageInfo:
    image = metadata.image_required
    axes: list[AxisInfo] = []
    for code, meaning in _AXES:
        bounds = image.dimension_bounds.get(code)
        indices = tuple(int(value) for value in image.dimension_indices.get(code, ()))
        start = int(bounds[0]) if bounds is not None else (min(indices) if indices else None)
        end = int(bounds[1]) if bounds is not None else (max(indices) + 1 if indices else None)
        coordinates: tuple[float, ...] = ()
        unit = None
        if code == "T" and image.posT is not None:
            coordinates = tuple(float(value) for value in image.posT)
            unit = "second"
        elif code == "Z" and image.posZ is not None:
            coordinates = tuple(float(value) for value in image.posZ)
            unit = "micrometer"
        axes.append(
            AxisInfo(
                code=code,
                meaning=meaning,
                size=getattr(image, f"Size{code}"),
                start=start,
                end=end,
                valid_indices=indices,
                unit=unit,
                coordinates=coordinates,
            )
        )
    first_scene_shape = None
    if image.SizeY_scene is not None and image.SizeX_scene is not None:
        first_scene_shape = (int(image.SizeY_scene), int(image.SizeX_scene))
    bounding_box = (
        metadata.bbox_required.total_bounding_box_no_pyramid or metadata.bbox_required.total_bounding_box or {}
    )
    return ImageInfo(
        dimension_order=metadata.sb_dimstring or "".join(axis.code for axis in axes if axis.size is not None),
        axes=tuple(axes),
        first_scene_shape_yx=first_scene_shape,
        total_bounding_box={str(key): (int(value[0]), int(value[1])) for key, value in bounding_box.items()},
        scene_shape_is_consistent=metadata.scene_shape_is_consistent,
        scene_shape_tolerance_px=metadata.scene_shape_tolerance,
    )


def _build_channels(metadata: CziMetadata) -> tuple[ChannelInfo, ...]:
    source = metadata.channelinfo_required
    channel_count = max(
        metadata.image_required.SizeC or 0,
        len(source.names),
        len(source.dyes),
        len(source.colors),
        len(source.pixeltypes),
    )
    channels = []
    for index in range(channel_count):
        limits = _at(source.clims, index)
        channels.append(
            ChannelInfo(
                index=index,
                name=_at(source.names, index),
                dye=_at(source.dyes, index),
                dye_short=_at(source.dyes_short, index),
                description=_at(source.channel_descriptions, index),
                color=_at(source.colors, index),
                black_point=float(limits[0]) if limits is not None and len(limits) > 0 else None,
                white_point=float(limits[1]) if limits is not None and len(limits) > 1 else None,
                gamma=_optional_float(_at(source.gamma, index)),
                pixel_type=source.pixeltypes.get(index) or (metadata.pixeltypes or {}).get(index),
                is_rgb=source.isRGB.get(index) if index in source.isRGB else (metadata.isRGB or {}).get(index),
            )
        )
    return tuple(channels)


def _build_instrument(metadata: CziMetadata) -> InstrumentInfo:
    microscope_source = metadata.microscope
    microscope = MicroscopeInfo(
        id=getattr(microscope_source, "Id", None),
        name=getattr(microscope_source, "Name", None),
        system=getattr(microscope_source, "System", None),
    )
    detector_source = metadata.detector
    detector_count = _parallel_count(
        detector_source,
        "Id",
        "name",
        "model",
        "modeltype",
        "gain",
        "zoom",
        "amplificationgain",
    )
    detectors = tuple(
        DetectorInfo(
            index=index,
            id=_source_at(detector_source, "Id", index),
            name=_source_at(detector_source, "name", index),
            model=_source_at(detector_source, "model", index),
            model_type=_source_at(detector_source, "modeltype", index),
            gain=_optional_float(_source_at(detector_source, "gain", index)),
            zoom=_optional_float(_source_at(detector_source, "zoom", index)),
            amplification_gain=_optional_float(_source_at(detector_source, "amplificationgain", index)),
        )
        for index in range(detector_count)
    )
    objective_source = metadata.objective
    objective_count = _parallel_count(objective_source, "Id", "name", "model", "immersion", "NA", "objmag")
    objectives = tuple(
        ObjectiveInfo(
            index=index,
            id=_source_at(objective_source, "Id", index),
            name=_source_at(objective_source, "name", index),
            model=_source_at(objective_source, "model", index),
            immersion=_source_at(objective_source, "immersion", index),
            numerical_aperture=_optional_float(_source_at(objective_source, "NA", index)),
            nominal_magnification=_optional_float(_source_at(objective_source, "objmag", index)),
            tube_lens_magnification=_optional_float(_source_at(objective_source, "tubelensmag", index)),
            total_magnification=_optional_float(_source_at(objective_source, "totalmag", index)),
        )
        for index in range(objective_count)
    )
    scale = metadata.scale_required
    has_scale = bool(getattr(metadata.czi_box, "has_scale", False))
    scaling = ScalingInfo(
        x=scale.X,
        y=scale.Y,
        z=scale.Z,
        unit=scale.unit,
        value_source="CZI Scaling.Items.Distance" if has_scale else "unavailable",
    )
    return InstrumentInfo(microscope=microscope, detectors=detectors, objectives=objectives, scaling=scaling)


def _build_sample(metadata: CziMetadata) -> SampleInfo:
    source = metadata.sample
    if source is None:
        return SampleInfo(None, None, None, None, None, None, "micrometer")
    carrier, carrier_source = _model_dict(source.sample_carrier)
    specimen, specimen_source = _model_dict(source.specimen)
    return SampleInfo(
        carrier=carrier,
        carrier_source=carrier_source,
        specimen=specimen,
        specimen_source=specimen_source,
        image_stage_x=source.image_stageX,
        image_stage_y=source.image_stageY,
        stage_unit="micrometer",
    )


def _build_hcs(metadata: CziMetadata) -> HcsInfo:
    declared = metadata.hcs_declared
    stored = metadata.hcs
    return HcsInfo(
        detected=bool(metadata.hcs_status.detected),
        detection_reason=metadata.hcs_status.reason,
        declared_plate=declared,
        stored_plate=stored,
        stored_scene_indices=tuple(int(value) for value in metadata.stored_scene_indices),
        declared_field_count=_field_count(declared),
        stored_field_count=_field_count(stored),
    )


def _build_attachments(metadata: CziMetadata) -> AttachmentInfo:
    source = metadata.attachments
    return AttachmentInfo(
        has_label=bool(source and source.has_label),
        has_preview=bool(source and source.has_preview),
        has_prescan=bool(source and source.has_prescan),
        names=tuple(source.names) if source else (),
    )


def _build_quality(
    metadata: CziMetadata,
    image: ImageInfo,
    channels: tuple[ChannelInfo, ...],
    hcs: HcsInfo,
    scaling: ScalingInfo,
) -> QualityInfo:
    issues: list[QualityIssue] = []
    if not hcs.detected:
        issues.append(QualityIssue("warning", "HCS_NOT_DETECTED", hcs.detection_reason, "hcs"))
    if hcs.declared_field_count > hcs.stored_field_count:
        missing = hcs.declared_field_count - hcs.stored_field_count
        issues.append(
            QualityIssue(
                "warning",
                "DECLARED_FIELD_NOT_STORED",
                f"{missing} XML-declared field(s) have no stored layer-0 scene.",
                "hcs.stored_plate",
            )
        )
    if hcs.stored_plate is not None:
        for well in hcs.stored_plate.wells:
            for field in well.fields:
                if field.position_conflict:
                    issues.append(
                        QualityIssue(
                            "warning",
                            "FIELD_POSITION_CONFLICT",
                            f"Field {field.id!r} has conflicting subblock positions.",
                            f"hcs.stored_plate.wells.{well.canonical_name}.fields.{field.field_index}",
                        )
                    )
    for axis in image.axes:
        if axis.start is not None and axis.end is not None and axis.valid_indices:
            if len(axis.valid_indices) != axis.end - axis.start:
                issues.append(
                    QualityIssue(
                        "warning",
                        "SPARSE_DIMENSION",
                        f"Axis {axis.code} has sparse stored indices.",
                        f"image.axes.{axis.code}",
                    )
                )
    if not image.scene_shape_is_consistent:
        issues.append(
            QualityIssue(
                "warning",
                "INCONSISTENT_SCENE_SHAPE",
                "Stored scenes do not have a consistent Y/X shape.",
                "image",
            )
        )
    expected_channels = metadata.image_required.SizeC or 0
    if len(channels) < expected_channels or any(channel.name is None for channel in channels):
        issues.append(
            QualityIssue(
                "warning",
                "MISSING_CHANNEL_METADATA",
                "One or more stored channels lack descriptive metadata.",
                "channels",
            )
        )
    if scaling.value_source == "unavailable":
        issues.append(
            QualityIssue(
                "warning",
                "SCALING_VALUE_DEFAULTED",
                "Physical scaling metadata is unavailable.",
                "instrument.scaling",
            )
        )
    return QualityInfo(status="ok" if not issues else "warning", issues=tuple(issues))


def _validate_axes(axes: tuple[AxisInfo, ...]) -> None:
    codes = [axis.code for axis in axes]
    if len(codes) != len(set(codes)):
        raise ValueError("Image axis codes must be unique.")
    for axis in axes:
        if (axis.start is None) != (axis.end is None):
            raise ValueError(f"Axis {axis.code} must define both start and end bounds.")
        if axis.start is not None and axis.end is not None:
            if axis.start > axis.end:
                raise ValueError(f"Axis {axis.code} has inverted bounds.")
            if any(index < axis.start or index >= axis.end for index in axis.valid_indices):
                raise ValueError(f"Axis {axis.code} has a valid index outside its bounds.")


def _validate_plate(plate: CziPlate | None) -> None:
    if plate is None:
        return
    well_ids = [well.id for well in plate.wells]
    well_names = [well.canonical_name for well in plate.wells]
    if len(well_ids) != len(set(well_ids)) or len(well_names) != len(set(well_names)):
        raise ValueError("Plate well IDs and canonical names must be unique.")
    field_ids: list[str] = []
    scene_indices: list[int] = []
    for well in plate.wells:
        _, row_index, column_index = normalize_well_name(well.canonical_name)
        if (well.row_index, well.column_index) != (row_index, column_index):
            raise ValueError(f"Well {well.canonical_name!r} has inconsistent normalized indices.")
        if [field.field_index for field in well.fields] != list(range(len(well.fields))):
            raise ValueError(f"Well {well.canonical_name!r} has non-contiguous field indices.")
        field_ids.extend(field.id for field in well.fields)
        scene_indices.extend(field.scene_index for field in well.fields)
    if len(field_ids) != len(set(field_ids)):
        raise ValueError("Field IDs must be unique within a plate view.")
    if len(scene_indices) != len(set(scene_indices)):
        raise ValueError("Scene indices must be unique within a plate view.")


def _validate_stored_fields(hcs: HcsInfo) -> None:
    if hcs.declared_plate is None or hcs.stored_plate is None:
        return
    declared = {field.scene_index for well in hcs.declared_plate.wells for field in well.fields}
    stored = {field.scene_index for well in hcs.stored_plate.wells for field in well.fields}
    if not stored.issubset(declared):
        raise ValueError("Every stored field must exist in the declared HCS hierarchy.")
    if hcs.declared_field_count != len(declared) or hcs.stored_field_count != len(stored):
        raise ValueError("HCS field counts do not match their plate views.")


def _validate_finite_values(value: Any, path: str = "document") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite floating-point value at {path}.")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _validate_finite_values(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_finite_values(child, f"{path}[{index}]")


def _to_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return _to_json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _to_json_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_value(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return _to_json_value(value.item())
    raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable.")


def _drop_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _drop_none(child) for key, child in value.items() if child is not None}
    if isinstance(value, list):
        return [_drop_none(child) for child in value]
    return value


def _model_dict(value: Any) -> tuple[dict[str, Any] | None, str | None]:
    if value is None:
        return None, None
    if hasattr(value, "model_dump"):
        payload = value.model_dump(mode="json")
        model_source = type(value).__name__
    elif isinstance(value, Mapping):
        payload = dict(value)
        model_source = None
    else:
        raise TypeError(f"Unsupported sample metadata type {type(value).__name__}.")
    normalized = _to_json_value(payload)
    if not isinstance(normalized, dict):
        raise TypeError("Sample metadata must serialize to an object.")
    return normalized, model_source


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _field_count(plate: CziPlate | None) -> int:
    return sum(len(well.fields) for well in plate.wells) if plate is not None else 0


def _parallel_count(source: Any, *names: str) -> int:
    if source is None:
        return 0
    return max((len(getattr(source, name, ())) for name in names), default=0)


def _source_at(source: Any, name: str, index: int) -> Any:
    return _at(getattr(source, name, ()), index) if source is not None else None


def _at(values: Any, index: int) -> Any:
    if values is None or index >= len(values):
        return None
    return values[index]


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _validate_units(image: ImageInfo, scaling: ScalingInfo) -> None:
    for axis in image.axes:
        if axis.coordinates and axis.unit is None:
            raise ValueError(f"Axis {axis.code} has coordinates but no explicit unit.")
    if any(value is not None for value in (scaling.x, scaling.y, scaling.z)) and scaling.unit is None:
        raise ValueError("Physical scaling values require an explicit unit.")


def _validate_extensions(extensions: Mapping[str, Any]) -> None:
    try:
        _to_json_value(dict(extensions))
    except TypeError as error:
        raise ValueError(f"extensions must contain only JSON-native values: {error}") from error


_ADAPTER: "TypeAdapter[CziHcsDocument] | None" = None


def _document_adapter() -> "TypeAdapter[CziHcsDocument]":
    global _ADAPTER
    if _ADAPTER is None:
        _ADAPTER = TypeAdapter(CziHcsDocument)
    return _ADAPTER
