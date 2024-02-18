import numpy as np

def min_max_normalization(img, name,max_val=None,verbose:bool=False):
    if max_val is None:
        max_val = np.iinfo(img.dtype).max

    img = (img - img.min()) / (img.max() - img.min()) * max_val

    if verbose: print(f'[info]: (preprocessing) Normalizing {name} with max value of {max_val}')

    return img