"""Provide all functionality for whitening an image"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from pylibCZIrw import czi as pyczi
import xmltodict
from tqdm.contrib.itertools import product  # type: ignore  # No stubs provided in this package
from scipy.fft import fft2, ifft2  # pylint: disable=no-name-in-module # type: ignore


def _get_image_dimensions(czidoc: pyczi.CziReader) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    """Determine the image dimensions.

    Arguments:
        czidoc: A CZI document reader.

    Returns:
        The s, t, z and c dimension in that order.
    """

    def _safe_get(key: str) -> Optional[int]:
        try:
            extracted_value: Tuple[int, int] = czidoc.total_bounding_box[key]
            return extracted_value[1] - extracted_value[0]
        except KeyError:
            return None

    size_s = len(czidoc.scenes_bounding_rectangle) if czidoc.scenes_bounding_rectangle else None
    size_t = _safe_get("T")
    size_z = _safe_get("Z")
    size_c = _safe_get("C")

    return size_s, size_t, size_z, size_c


def _get_scale(raw_metadata: Dict[Any, Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Determine the scaling coefficients for each spatial dimension.

    Arguments:
        raw_metadata: The CZI meta-data to derive the scaling coefficients from.

    Returns:
        The scaling coefficients in x, y and z direction in that order.
    """

    def _safe_get(distances_: List[Dict[Any, Any]], idx: int) -> Optional[float]:
        try:
            return float(distances_[idx]["Value"]) if distances_[idx]["Value"] is not None else None
        except IndexError:
            return None

    try:
        distances = raw_metadata["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"]
    except KeyError:
        return None, None, None

    scale_x = _safe_get(distances, 0)
    scale_y = _safe_get(distances, 1)
    scale_z = _safe_get(distances, 2)

    return scale_x, scale_y, scale_z


def _whitening(image: np.ndarray, order: int, spatial_axis: int) -> np.ndarray:
    """Applies a whitening filter on a given image in a given direction.

    The image is transformed to frequency space and the resulting profile by a polynomial of a given oder.
    The lowest frequencies are subsequently filtered out by setting the first coefficient of the calculated polynomial
    to 0 and sampling low-frequency correction values from the resulting polynomial.

    Arguments:
        image: The image to be whitened.
        order: The order of the polynomial to approximate the profile in frequency space.
        spatial_axis: The direction in which to apply whitening.
            0 means vertical direction and 1 means horizontal direction.

    Returns:
        The whitened image.

    Raises:
        ValueError: If whitening should be applied in at least one direction and the image does not
        have exactly three dimensions.
    """
    if len(image.shape) != 3:
        raise ValueError("The input image must have exactly three dimensions.")

    unselected_spatial_axis = 0 if spatial_axis == 1 else 1
    channels_whitened = []

    for channel in np.moveaxis(image, -1, 0):
        # do the fourier transform
        fourier = (fft2(channel.astype(np.float32), workers=-1)).astype(np.complex64)

        # split into magnitudes and phases
        mags = np.absolute(fourier)
        phase = np.angle(fourier)

        # average horizontal log profile
        mags_half_unselected_spatial_axis = mags.shape[unselected_spatial_axis] // 2
        profile_start = [0, 0]
        profile_start[unselected_spatial_axis] = mags_half_unselected_spatial_axis - 10
        profile_end = [list(mags.shape)[0], list(mags.shape)[1]]
        profile_end[unselected_spatial_axis] = mags_half_unselected_spatial_axis + 10
        profile = np.log(mags[profile_start[0]: profile_end[0], profile_start[1]: profile_end[1]])
        profile = np.average(profile, axis=unselected_spatial_axis)

        # do a polynomial fit
        xvals = np.arange(0, profile.shape[0])
        polynomial: np.polynomial.Polynomial = np.polynomial.Polynomial.fit(xvals, profile, deg=order, full=False)
        polynomial = polynomial.convert()
        # Set last polynomial coefficient to 0(constant)
        polynomial_approx = np.polynomial.Polynomial(
            [0.0] + list(polynomial.coef[1:]),
            domain=polynomial.domain,
            window=polynomial.window,
        )

        corrvals = polynomial_approx(xvals)
        corr_vals_l: np.ndarray = np.expand_dims(corrvals, axis=unselected_spatial_axis)
        corr_vals_l = np.repeat(
            corr_vals_l,
            mags.shape[unselected_spatial_axis],
            axis=unselected_spatial_axis,
        )

        # subtract fitted average profile from magnitude
        corrected_mags = np.exp(np.log(mags) - corr_vals_l)

        corrected_i = corrected_mags * np.sin(phase)
        corrected_j = corrected_mags * np.cos(phase)
        corrected_j = corrected_j.astype(np.complex64)
        corrected = corrected_j + (1j * corrected_i)
        dataset_corrected = ifft2(corrected, workers=-1)

        # convert to float32
        channels_whitened.append((np.real(dataset_corrected)).astype(np.float32))

    return np.stack(channels_whitened, axis=-1).astype(image.dtype)


def whitening(
    image: np.ndarray,
    horizontal: bool,
    vertical: bool,
    order: int = 8,
) -> np.ndarray:
    """Function to remove correlation for the noise to create "white noise"

    Args:
        image: 2D NumPy array with the pixel values
        horizontal: Whether to apply whitening horizontally
        vertical: Whether to apply whitening vertically
        order: The order of the polynomial to approximate the profile in frequency space.

    Returns:
         The whitened image.

    """
    whitened = _whitening(image, order, 1) if horizontal else image
    whitened = _whitening(whitened, order, 0) if vertical else whitened
    return whitened


def execute(
    input_path: str,
    output_path: str,
    horizontal: bool,
    vertical: bool,
) -> None:
    """Execute whitening for an entire CZI image.

    Arguments:
        input_path: The path of the input image.
        output_path: The path in which to store the whitened image.
        horizontal: Whether to apply horizontal whitening.
        vertical: Whether to apply vertical whitening.
    """
    # open new CZI document for writing
    with pyczi.create_czi(output_path) as czidoc_w:
        # open the original CZI document to read 2D image planes
        with pyczi.open_czi(input_path) as czidoc_r:
            metadata_parsed = xmltodict.parse(czidoc_r.raw_metadata)
            size_s, size_t, size_z, size_c = _get_image_dimensions(czidoc_r)
            scale_x, scale_y, scale_z = _get_scale(metadata_parsed)

            # read array for the scene
            for s, t, z, c in product(
                range(size_s) if size_s is not None else [None],
                range(size_t) if size_t is not None else [None],
                range(size_z) if size_z is not None else [None],
                range(size_c) if size_c is not None else [None],
            ):
                plane = czidoc_r.read(
                    plane={k: v for k, v in {"T": t, "Z": z, "C": c}.items() if v is not None},
                    scene=s,
                )
                plane_whitened = whitening(plane, horizontal=horizontal, vertical=vertical)
                write_location = czidoc_r.scenes_bounding_rectangle[s] if s is not None else (0, 0)
                czidoc_w.write(
                    plane_whitened,
                    plane={k: v for k, v in {"T": t, "Z": z, "C": c}.items() if v is not None},
                    scene=s if s is not None else 0,
                    location=write_location,
                )

            # Write scaling the new czi
            # Adhering to the ZISRAW specification the unit is provided in meters
            czidoc_w.write_metadata(
                scale_x=scale_x,
                scale_y=scale_y,
                scale_z=scale_z,
            )
