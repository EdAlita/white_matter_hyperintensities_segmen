import numpy
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


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
             verbose: int =1):
    """Pad or crop the size of an input volume to a reference shape
    Args:
        arr (numpy.ndarray):  array to be map
        base_shape (3D ref size) : 3D size of the reference size
        verbose : (int) Verbosity, to turn of set to 0. Default is '1'
    Returns:
        final_arr (3D array) : 3D array containing with a shape defined by base_shape
    """

    if verbose > 0:
        print('Volume will be resize from %s to %s ' % (arr.shape, base_shape))

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


if __name__ == '__main__':
    
    img = nib.load('/localmount/volume-hd/users/uline/data_sets/CVD/0a3a5d79-a26c-4aea-99f4-892751d2baa7/FLAIR.nii.gz')
    canonical = nib.as_closest_canonical(img)
    flair = canonical.get_fdata()
    print(type(flair))
    print(nib.aff2axcodes(canonical.affine))
    print(type([256, 256, 256]))

    new_size = map_size(flair, [256, 256, 256], verbose=1)
    plt.imshow(new_size[100, :, :], cmap='gray')
    plt.show()

    resize_img = nib.Nifti1Image(new_size, img.affine, img.header)
    nib.save(resize_img, 'resize.nii.gz')
