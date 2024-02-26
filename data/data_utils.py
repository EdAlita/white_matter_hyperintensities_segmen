from numpy import typing as npt
import numpy as np
from typing import Tuple


def get_thick_slices(
        img_data: npt.NDArray,
        slice_thickness: int = 3
) -> np.ndarray:
    """
    Extract thick slices from the image.

    Feed slice_thickness preceding and succeeding slices to network,
    label only middle one.

    Parameters
    ----------
    img_data : npt.NDArray
        3D MRI image read in with nibabel.
    slice_thickness : int
        Number of slices to stack on top and below slice of interest (default=3).

    Returns
    -------
    np.ndarray
        Image data with the thick slices of the n-th axis appended into the n+1-th axis.
    """
    img_data_pad = np.pad(img_data, ((0, 0), (0, 0), (slice_thickness, slice_thickness)), mode="edge")
    from numpy.lib.stride_tricks import sliding_window_view

    # sliding_window_view will automatically create thick slices through a sliding window, but as this in only a view,
    # less memory copies are required
    return sliding_window_view(img_data_pad, 2 * slice_thickness + 1, axis=2)


def filter_blank_slices_thick(
        img_vol: npt.NDArray,
        label_vol: npt.NDArray,
        threshold: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter blank slices from the volume using the label volume.

    Parameters
    ----------
    img_vol : npt.NDArray
        Orig image volume.
    label_vol : npt.NDArray
        Label images (ground truth).
    threshold : int
        Threshold for number of pixels needed to keep slice (below = dropped). (Default value = 50).

    Returns
    -------
    filtered img_vol : np.ndarray
        [MISSING].
    label_vol : np.ndarray
        [MISSING].
    """
    # Get indices of all slices with more than threshold labels/pixels
    select_slices = np.sum(label_vol, axis=(0, 1)) > threshold

    # Retain only slices with more than threshold labels/pixels
    img_vol = img_vol[:, :, select_slices, :]
    label_vol = label_vol[:, :, select_slices]

    return img_vol, label_vol


def sagittal_transform_coronal(vol: np.ndarray, inverse: bool = False) -> np.ndarray:
    if inverse:
        return np.moveaxis(vol, [0, 1, 2], [1, 0, 2])
    else:
        return np.moveaxis(vol, [0, 1, 2], [1, 0, 2])


def sagittal_transform_axial(vol: np.ndarray, inverse: bool = False) -> np.ndarray:
    if inverse:
        return np.moveaxis(vol, [0, 1, 2], [2, 1, 0])
    else:
        return np.moveaxis(vol, [0, 1, 2], [1, 2, 0])