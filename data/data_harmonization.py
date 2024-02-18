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

def interpolation_volume(volume, original_spacing, target_spacing=[1,1,1], method='linear'):
    scale_factor = np.array(original_spacing) / np.array(target_spacing)
    new_shape = np.cell(np.array(volume.shape)* scale_factor).astype(int)

    z, y, x = np.mgrid[ 0:volume.shape[0],0:volume.shape[1],0:volume.shape[2]]

    z_new = 

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