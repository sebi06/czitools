"""Validate local OME-Zarr image and HCS stores.

Vendored (with light edits) from ``czi_omezarr_utils.validation`` in the
``omezarr_playground`` repository as part of czitools Stage 5.
"""

import logging
from pathlib import Path
from typing import Any, cast

import ngff_zarr as nz
import zarr
from ome_zarr_models.v05.image import Image
from ome_zarr_models.v05.plate import Plate
from pydantic import ValidationError

logger = logging.getLogger(__name__)


def validate_ome_zarr(path: str | Path) -> bool:
    """Validate a local OME-Zarr file against its declared OME-NGFF version.

    OME-NGFF 0.5 image and HCS layouts use ``ome-zarr-models`` Pydantic
    models. OME-NGFF 0.6 images are parsed by ``ngff-zarr``, whose reader
    understands the RFC-5 coordinate-system and transformation model.

    Args:
        path (Union[str, Path]): Path to the OME-Zarr directory or archive.

    Returns:
        bool: ``True`` if the file is valid, ``False`` otherwise.
    """
    path = str(path)
    try:
        group = zarr.open_group(path, mode="r")
        root_attrs = group.attrs.asdict()
        ome_attrs = root_attrs.get("ome", {})
        declared_version = ome_attrs.get("version") if isinstance(ome_attrs, dict) else None

        if declared_version in {"0.6", "0.6rc0", "0.6.dev4"}:
            nz.from_ngff_zarr(path)
            logger.info(f"Valid OME-ZARR {declared_version} image: {path}")
            return True

        if isinstance(ome_attrs, dict) and "plate" in ome_attrs:
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

            logger.info(f"Valid OME-ZARR HCS plate: {path}")
        else:
            Image.from_zarr(group)
            logger.info(f"Valid OME-ZARR image: {path}")

        return True

    except ValidationError as e:
        logger.error(f"Validation failed: {path}\n{e}")
        return False
    except Exception as e:
        logger.error(f"Error opening file: {e}")
        return False
