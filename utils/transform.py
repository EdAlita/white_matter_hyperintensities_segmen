import numpy as np


def axial_transform_coronal(vol: np.ndarray, inverse: bool = False) -> np.ndarray:
    if inverse:
        return np.moveaxis(vol, [0, 1, 2], [1, 0, 2])
    else:
        return np.moveaxis(vol, [0, 1, 2], [1, 0, 2])


def axial_transform_sagital(vol: np.ndarray, inverse: bool = False) -> np.ndarray:
    if inverse:
        return np.moveaxis(vol, [0, 1, 2], [2, 0, 1])
    else:
        return np.moveaxis(vol, [0, 1, 2], [1, 2, 0])


