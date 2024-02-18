"""
TO DO: 1 mm data interpolation
       Move to RAS
       256,256,256
       intesity rescale
       
       import numpy as np
       import torch
       import torch.nn.functional as F
"""""

import nibabel as nib
import pathlib as Path
import numpy as np

def trilinear_interpolation(volume, x, y, z):

    x0, y0, z0 = int(np.floor(x)), int(np.floor(y)), int(np.floor(z))
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1

    x0, x1 = np.clip([x0, x1], 0, volume.shape[2] - 1)
    y0, y1 = np.clip([y0, y1], 0, volume.shape[1] - 1)
    z0, z1 = np.clip([z0, z1], 0, volume.shape[0] - 1)

    xd, yd, zd = x - x0, y - y0, z - z0

    c00 = volume[z0, y0, x0] * (1 - xd) + volume[z0, y0, x1] * xd
    c01 = volume[z0, y1, x0] * (1 - xd) + volume[z0, y1, x1] * xd
    c10 = volume[z1, y0, x0] * (1 - xd) + volume[z1, y0, x1] * xd
    c11 = volume[z1, y1, x0] * (1 - xd) + volume[z1, y1, x1] * xd

    c0 = c00 * (1 - yd) + c01 * yd
    c1 = c10 * (1 - yd) + c11 * yd

    c = c0 * (1 - zd) + c1 * zd

    return c

def interpolate_volume(volume, original_spacing, target_spacing=[1, 1, 1], method='linear'):
    scale_factors = np.array(original_spacing) / np.array(target_spacing)
    new_shape = np.ceil(np.array(volume.shape) * scale_factors).astype(int)

    interpolated_volume = np.zeros(new_shape)

    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            for k in range(new_shape[2]):
                if method == 'nearest':
                    # Nearest neighbor interpolation
                    nearest_i, nearest_j, nearest_k = int(round(i / scale_factors[0])), int(round(j / scale_factors[1])), int(round(k / scale_factors[2]))
                    interpolated_volume[i, j, k] = volume[min(nearest_i, volume.shape[0] - 1), min(nearest_j, volume.shape[1] - 1), min(nearest_k, volume.shape[2] - 1)]
                elif method == 'linear':
                    # Trilinear interpolation
                    original_i, original_j, original_k = i / scale_factors[0], j / scale_factors[1], k / scale_factors[2]
                    interpolated_volume[i, j, k] = trilinear_interpolation(volume, original_k, original_j, original_i)
    
    return interpolated_volume

def define_size(mov_dim: np.array,
                ref_dim: np.array) -> [list, list]:
    """Calculate a new image size by duplicate the size of the bigger ones
    Args:
        mov_dim (np.array):  3D size of the input volume
        ref_dim (np.array) : 3D size of the reference size
    Returns:
        new_dim (list) : New array size
        borders (list) : border Index for mapping the old volume into the new one
    """
    new_dim = np.zeros(len(mov_dim), dtype=np.int16)
    borders = np.zeros((len(mov_dim), 2), dtype=np.int16)
    padd = [int(mov_dim[0] // 2), int(mov_dim[1] // 2), int(mov_dim[2] // 2)]

    for i in range(len(mov_dim)):
        new_dim[i] = int(max(2 * mov_dim[i], 2 * ref_dim[i]))
        borders[i, 0] = int(new_dim[i] // 2) - padd[i]
        borders[i, 1] = borders[i, 0] + mov_dim[i]

    return list(new_dim), borders


def map_size(arr: numpy.ndarray,
             base_shape: list,
             verbose: bool =False):
    """Pad or crop the size of an input volume to a reference shape
    Args:
        arr (numpy.ndarray):  array to be map
        base_shape (3D ref size) : 3D size of the reference size
        verbose : (bool) Verbosity, to turn of set to 0. Default is '1'
    Returns:
        final_arr (3D array) : 3D array containing with a shape defined by base_shape
    """

    if verbose: print('Volume will be resize from %s to %s ' % (arr.shape, base_shape))

    if list(arr.shape) != list(base_shape):

        new_shape, borders = define_size(np.array(arr.shape), np.array(base_shape))
        new_arr = np.zeros(new_shape)
        final_arr = np.zeros(base_shape)

        new_arr[borders[0, 0]:borders[0, 1], borders[1, 0]:borders[1, 1], borders[2, 0]:borders[2, 1]] = arr[:]

        middle_point = [int(new_arr.shape[0] // 2), int(new_arr.shape[1] // 2), int(new_arr.shape[2] // 2)]
        padd = [int(base_shape[0] / 2), int(base_shape[1] / 2), int(base_shape[2] / 2)]

        low_border = np.array((np.array(middle_point) - np.array(padd)), dtype=np.int16)
        high_border = np.array(np.array(low_border) + np.array(base_shape), dtype=np.int16)

        final_arr[:, :, :] = new_arr[low_border[0]:high_border[0],
                             low_border[1]:high_border[1],
                             low_border[2]:high_border[2]]

        return final_arr

    else:
        return arr

def data_harmonization(
        root_path: Path,
        verbose: bool = False
        ):

        if not root_path.is_dir():
            raise NotADirectoryError()

        for folder in sorted(root_path.iterdir()):
            if folder.is_dir():
                if verbose: print(f'Processing {folder.name}')
                for file in folder.glob('*.nii.gz'):

                    img = nib.load(file)

                    if nib.aff2axcodes(img.affine) != ('R', 'A', 'S'):
                        img = nib.as_closest_canonical(img)
                        if verbose: print(f'[info]: Converting {file.name} to RAS orientation')

                    img_array = img.get_fdata()



                    resize_img = nib.Nifti1Image(new_size, img.affine, img.header)
                    nib.save(resize_img, file)

if __name__ == '__main__':
    data_harmonization()