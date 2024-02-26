""""
TO DO: 1 mm data interpolation
       Move to RAS
       256,256,256
       intesity rescale
       
       import numpy as np
       import torch
       import torch.nn.functional as F
"""""
from nibabel.processing import resample_to_output
import numpy as np

def rescale_image(img_data):
    # Conform intensities
    src_min, scale = getscale(img_data, 0, 255)
    mapped_data = img_data
    if not img_data.dtype == np.dtype(np.uint8):
        if np.max(img_data) > 255:
            mapped_data = scalecrop(img_data, 0, 255, src_min, scale)

    new_data = np.uint8(np.rint(mapped_data))
    return new_data


def getscale(data, dst_min, dst_max, f_low=0.0, f_high=0.999):
    """
    Function to get offset and scale of image intensities to robustly rescale to range dst_min..dst_max.
    Equivalent to how mri_convert conforms images.

    :param np.ndarray data: Image data (intensity values)
    :param float dst_min: future minimal intensity value
    :param float dst_max: future maximal intensity value
    :param f_low: robust cropping at low end (0.0 no cropping)
    :param f_high: robust cropping at higher end (0.999 crop one thousandths of high intensity voxels)
    :return: returns (adjusted) src_min and scale factor
    """
    # get min and max from source
    src_min = np.min(data)
    src_max = np.max(data)

    if src_min < 0.0:
        sys.exit('ERROR: Min value in input is below 0.0!')

    # print("Input:    min: " + format(src_min) + "  max: " + format(src_max))

    if f_low == 0.0 and f_high == 1.0:
        return src_min, 1.0

    # compute non-zeros and total vox num
    nz = (np.abs(data) >= 1e-15).sum()
    voxnum = data.shape[0] * data.shape[1] * data.shape[2]

    # compute histogram
    histosize = 1000
    bin_size = (src_max - src_min) / histosize
    hist, bin_edges = np.histogram(data, histosize)

    # compute cummulative sum
    cs = np.concatenate(([0], np.cumsum(hist)))

    # get lower limit
    nth = int(f_low * voxnum)
    idx = np.where(cs < nth)

    if len(idx[0]) > 0:
        idx = idx[0][-1] + 1

    else:
        idx = 0

    src_min = idx * bin_size + src_min

    # print("bin min: "+format(idx)+"  nth: "+format(nth)+"  passed: "+format(cs[idx])+"\n")
    # get upper limit
    nth = voxnum - int((1.0 - f_high) * nz)
    idx = np.where(cs >= nth)

    if len(idx[0]) > 0:
        idx = idx[0][0] - 2

    else:
        print('ERROR: rescale upper bound not found')

    src_max = idx * bin_size + src_min
    # print("bin max: "+format(idx)+"  nth: "+format(nth)+"  passed: "+format(voxnum-cs[idx])+"\n")

    # scale
    if src_min == src_max:
        scale = 1.0

    else:
        scale = (dst_max - dst_min) / (src_max - src_min)

    # print("rescale:  min: " + format(src_min) + "  max: " + format(src_max) + "  scale: " + format(scale))
    return src_min, scale


def scalecrop(data, dst_min, dst_max, src_min, scale):
    """
    Function to crop the intensity ranges to specific min and max values

    :param np.ndarray data: Image data (intensity values)
    :param float dst_min: future minimal intensity value
    :param float dst_max: future maximal intensity value
    :param float src_min: minimal value to consider from source (crops below)
    :param float scale: scale value by which source will be shifted
    :return: scaled Image data array
    """
    data_new = dst_min + scale * (data - src_min)

    # clip
    data_new = np.clip(data_new, dst_min, dst_max)
    # print("Output:   min: " + format(data_new.min()) + "  max: " + format(data_new.max()))

    return data_new

def interpolate_volume(volume, target_spacing = (1.0, 1.0, 1.0), interpolation = ''):

    if interpolation == 'nearest':
        return resample_to_output(volume,voxel_sizes=target_spacing,mode='nearest')
    else:
        return resample_to_output(volume,voxel_sizes=target_spacing)

def define_size(mov_dim:np.array,ref_dim: np.array) -> [list, list]:
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


def map_size(arr: np.ndarray,base_shape: list,verbose: bool = False):
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
