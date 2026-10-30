"""Internal NGFF capabilities and version-independent axis descriptions.

Draft versions deliberately have no profile until their exact metadata model,
schemas, and reader compatibility have been verified.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import warnings

import zarr

DEFAULT_NGFF_VERSION = "0.6"
TCZYX = ("t", "c", "z", "y", "x")


@dataclass(frozen=True)
class AxisDescription:
    """Describe an axis without imposing an NGFF version's layout restrictions."""

    name: str
    type: str | None
    unit: str | None = None
    discrete: bool | None = None


def describe_axes(
    dims: Sequence[str],
    units: Mapping[str, str] | None = None,
    axis_types: Mapping[str, str | None] | None = None,
    discrete: Mapping[str, bool] | None = None,
) -> tuple[AxisDescription, ...]:
    """Describe known axes explicitly, leaving unspecified custom types unknown.

    Args:
        dims (Sequence[str]): Axis names in actual array dimension order.
        units (Mapping[str, str] | None): Physical units, when known.
        axis_types (Mapping[str, str | None] | None): Explicit type overrides.
        discrete (Mapping[str, bool] | None): Explicit interpolation semantics.

    Returns:
        tuple[AxisDescription, ...]: Ordered, version-independent descriptions.
    """
    types: dict[str, str | None] = {"t": "time", "c": "channel", "z": "space", "y": "space", "x": "space"}
    types.update(axis_types or {})
    units = units or {}
    discrete = discrete or {}
    return tuple(
        AxisDescription(
            name=name,
            type=types.get(name),
            unit=units.get(name),
            discrete=discrete.get(name, {"time": False, "space": False, "channel": True}.get(types.get(name))),
        )
        for name in dims
    )


@dataclass(frozen=True)
class VersionProfile:
    """Keep supported storage, validation, and image-axis limits in one place."""

    version: str
    zarr_format: int
    writer_backends: tuple[str, ...]
    validation_backend: str
    supports_archives: bool
    min_dimensions: int = 2
    max_dimensions: int = 5
    min_spatial_axes: int = 2
    max_spatial_axes: int = 3
    max_time_axes: int = 1
    max_other_axes: int = 1
    ordered_axes: bool = True

    def validate_axes(self, axes: Sequence[AxisDescription], shape: Sequence[int]) -> None:
        """Reject layouts that cannot be represented by this export profile."""
        names = [axis.name for axis in axes]
        if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
            raise ValueError("NGFF axis names must be non-empty and unique.")
        if len(axes) != len(shape):
            raise ValueError("NGFF axes must match the array dimensionality and order.")
        if not self.min_dimensions <= len(axes) <= self.max_dimensions:
            raise ValueError(
                f"OME-NGFF {self.version} requires {self.min_dimensions}-{self.max_dimensions} dimensions."
            )
        spatial = sum(axis.type == "space" for axis in axes)
        temporal = sum(axis.type == "time" for axis in axes)
        other = len(axes) - spatial - temporal
        if not self.min_spatial_axes <= spatial <= self.max_spatial_axes:
            raise ValueError(
                f"OME-NGFF {self.version} requires {self.min_spatial_axes}-{self.max_spatial_axes} spatial axes."
            )
        if temporal > self.max_time_axes or other > self.max_other_axes:
            raise ValueError(f"OME-NGFF {self.version} permits at most one time and one channel/custom axis.")
        order = [0 if axis.type == "time" else 2 if axis.type == "space" else 1 for axis in axes]
        if self.ordered_axes and order != sorted(order):
            raise ValueError(f"OME-NGFF {self.version} requires time, channel/custom, then spatial axis order.")


_PROFILES = {
    "0.4": VersionProfile("0.4", 2, ("ngff-zarr",), "ngff-zarr", False),
    "0.5": VersionProfile("0.5", 3, ("ngff-zarr", "ome-zarr-py"), "ome-zarr-models", True),
    "0.6": VersionProfile("0.6", 3, ("ngff-zarr", "ome-zarr-py"), "ngff-zarr", True),
}
_READ_ALIASES = {"0.6rc0": "0.6", "0.6.dev4": "0.6"}


def get_version_profile(version: object) -> VersionProfile:
    """Resolve only supported versions or explicitly recognized read aliases."""
    if isinstance(version, str):
        profile = _PROFILES.get(_READ_ALIASES.get(version, version))
        if profile is not None:
            return profile
    raise ValueError(f"Unsupported OME-NGFF version {version!r}; draft versions are not enabled.")


def get_export_profile(version: str, backend: str, *, hcs: bool = False, archive: bool = False) -> VersionProfile:
    """Check exact writer capabilities before any output is created or replaced."""
    supported = [profile for profile in _PROFILES.values() if backend in profile.writer_backends]
    profile = next((profile for profile in supported if profile.version == version), None)
    if profile is None:
        kind = "HCS" if hcs else "image"
        versions = ", ".join(profile.version for profile in supported)
        raise ValueError(
            f"{backend} {kind} writing currently supports OME-NGFF versions {versions}; got {version!r}. "
            "Draft versions are not enabled."
        )
    if archive and not profile.supports_archives:
        raise ValueError(f"RFC-9 .ozx output is not supported for OME-NGFF {version}.")
    return profile


def validate_hierarchy_metadata(group: zarr.Group, version: str) -> None:
    """Check version, axes, and display compatibility without loading pixels.

    Args:
        group (zarr.Group): Open image, plate, well, or metadata subgroup.
        version (str): Exact version declared by the hierarchy root.

    Raises:
        ValueError: Metadata mixes versions or exceeds the selected profile.
    """
    profile = get_version_profile(version)
    attributes = group.attrs.asdict()
    metadata = attributes.get("ome") if profile.zarr_format == 3 else attributes
    if metadata is not None:
        if not isinstance(metadata, dict):
            raise ValueError(f"Invalid NGFF metadata at {group.path!r}.")
        if profile.zarr_format == 3 and metadata.get("version") != version:
            raise ValueError(f"Mixed or missing NGFF version at {group.path!r}; expected {version!r}.")
        if group.metadata.zarr_format != profile.zarr_format:
            raise ValueError(f"OME-NGFF {version} requires Zarr {profile.zarr_format}.")
        for multiscale in metadata.get("multiscales", []):
            if profile.zarr_format == 2 and multiscale.get("version") != version:
                raise ValueError(f"Mixed NGFF versions at {group.path!r}.")
            systems = (
                multiscale.get("coordinateSystems", [])
                if profile.version == "0.6"
                else [{"axes": multiscale.get("axes", [])}]
            )
            for dataset in multiscale.get("datasets", []):
                array = group[dataset["path"]]
                if not isinstance(array, zarr.Array):
                    raise ValueError("A multiscale dataset must reference a Zarr array.")
                for system in systems:
                    axes = [
                        AxisDescription(axis["name"], axis.get("type"), axis.get("unit"), axis.get("discrete"))
                        for axis in system.get("axes", [])
                    ]
                    profile.validate_axes(axes, array.shape)
        if "omero" in metadata:
            for channel in metadata["omero"]["channels"]:
                color = channel.get("color")
                window = channel.get("window", {})
                if not isinstance(color, str) or len(color) != 6 or set(color.lower()) - set("0123456789abcdef"):
                    raise ValueError("Channel metadata requires a six-digit RGB color.")
                if not isinstance(window, dict) or not {"min", "max", "start", "end"} <= window.keys():
                    raise ValueError("Channel metadata requires complete min/max/start/end window values.")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Object at .* is not recognized as a component of a Zarr hierarchy\.",
            category=zarr.errors.ZarrUserWarning,
        )
        children = tuple(group.groups())
    for _, child in children:
        validate_hierarchy_metadata(child, version)
