"""Validate local OME-Zarr image and HCS stores.

Vendored (with light edits) from ``czi_omezarr_utils.validation`` in the
``omezarr_playground`` repository as part of czitools Stage 5.
"""

import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import ngff_zarr as nz
import zarr
from ome_zarr_models.v05.image import Image
from ome_zarr_models.v05.plate import Plate
from pydantic import ValidationError

from ._spec import get_version_profile, validate_hierarchy_metadata

logger = logging.getLogger(__name__)


def validate_ome_zarr(path: str | Path) -> bool:
    """Validate a local OME-Zarr file against its declared OME-NGFF version.

    Explicit version profiles select the schema validator. Unknown versions,
    mixed-version hierarchies, and layouts outside the selected profile fail
    validation rather than falling back to an older metadata model.

    Args:
        path (Union[str, Path]): Path to the OME-Zarr directory or archive.

    Returns:
        bool: ``True`` if the file is valid, ``False`` otherwise.
    """
    path = str(path)
    try:
        store_context = (
            zarr.storage.ZipStore(path, mode="r") if Path(path).suffix.lower() == ".ozx" else nullcontext(path)
        )
        with store_context as store:
            group = zarr.open_group(store, mode="r")
            root_attrs = group.attrs.asdict()
            ome_attrs = root_attrs.get("ome", root_attrs)
            declared_version = ome_attrs.get("version") if isinstance(ome_attrs, dict) else None
            if declared_version is None and isinstance(ome_attrs, dict):
                multiscales = ome_attrs.get("multiscales", [])
                if multiscales:
                    declared_version = multiscales[0].get("version")
                elif "plate" in ome_attrs:
                    declared_version = ome_attrs["plate"].get("version")
            profile = get_version_profile(declared_version)
            validate_hierarchy_metadata(group, declared_version)

            if isinstance(ome_attrs, dict) and "plate" in ome_attrs:
                if profile.validation_backend == "ngff-zarr":
                    nz.from_hcs_zarr(path, validate=True)
                else:
                    plate = Plate.model_validate(ome_attrs["plate"])
                    for well in plate.wells:
                        well_group = group[well.path]
                        if not isinstance(well_group, zarr.Group):
                            raise TypeError(f"Expected well group at {well.path}, got {type(well_group).__name__}")

                        well_ome_attrs = well_group.attrs.asdict().get("ome", {})
                        well_attrs: Any = well_ome_attrs.get("well", {}) if isinstance(well_ome_attrs, dict) else {}
                        image_entries = well_attrs.get("images", []) if isinstance(well_attrs, dict) else []

                        for image_info in image_entries:
                            if not isinstance(image_info, dict) or "path" not in image_info:
                                continue
                            image_group = well_group[cast(str, image_info["path"])]
                            if not isinstance(image_group, zarr.Group):
                                raise TypeError(
                                    f"Expected image group at {well.path}/{image_info['path']}, "
                                    f"got {type(image_group).__name__}"
                                )
                            Image.from_zarr(image_group)
                logger.info("Valid OME-ZARR %s HCS plate: %s", declared_version, path)
            else:
                if profile.validation_backend == "ngff-zarr":
                    nz.from_ngff_zarr(path, validate=True)
                else:
                    Image.from_zarr(group)
                logger.info("Valid OME-ZARR %s image: %s", declared_version, path)

        return True

    except ValidationError as e:
        logger.error(f"Validation failed: {path}\n{e}")
        return False
    except Exception as e:
        logger.error(f"Error opening file: {e}")
        return False
