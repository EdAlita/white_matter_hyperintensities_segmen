from pathlib import Path
import os
import nibabel as nib
import numpy as np


import shutil
def re_name_data(
        root_path: Path,
        verbose: bool = False
):
    if not root_path.is_dir(): raise NotADirectoryError()
    for folder in sorted(root_path.iterdir()):
        if folder.is_dir():
            for file in folder.iterdir():
                if 'label' in file.name or 'lesion' in file.name:
                    if verbose: print(folder / file.name, folder / 'lesion.nii.gz')
                    os.rename(folder / file.name, folder / 'lesion.nii.gz')
                    img = nib.load(folder / folder / 'lesion.nii.gz')
                    gt_seg = np.asarray(img.get_fdata(), dtype=np.uint8)
                    img.header.set_data_dtype(np.uint8)
                    img_out = nib.Nifti1Image(gt_seg, img.affine, img.header)
                    nib.save(img_out, folder / 'lesion.nii.gz')
                elif 'FLAIR' in file.name:
                    if verbose: print(folder / file.name, folder / 'FLAIR.nii.gz')
                    os.rename(folder / file.name, folder / 'FLAIR.nii.gz')
                    img = nib.load(folder / folder / 'FLAIR.nii.gz')
                    gt_seg = np.asarray(img.get_fdata(), dtype=np.uint8)
                    img.header.set_data_dtype(np.uint8)
                    img_out = nib.Nifti1Image(gt_seg, img.affine, img.header)
                    nib.save(img_out, folder / 'FLAIR.nii.gz')
                elif 'T1' in file.name:
                    if verbose: print(folder / file.name, folder / 'T1.nii.gz')
                    os.rename(folder / file.name, folder / 'T1.nii.gz')
                    img = nib.load(folder / folder / 'T1.nii.gz')
                    gt_seg = np.asarray(img.get_fdata(), dtype=np.uint8)
                    img.header.set_data_dtype(np.uint8)
                    img_out = nib.Nifti1Image(gt_seg, img.affine, img.header)
                    nib.save(img_out, folder / 'T1.nii.gz')
                elif 'T2' in file.name:
                    if verbose: print(folder / file.name, folder / 'T2.nii.gz')
                    os.rename(folder / file.name, folder / 'T2.nii.gz')
                    img = nib.load(folder / folder / 'T2.nii.gz')
                    gt_seg = np.asarray(img.get_fdata(), dtype=np.int16)
                    img.header.set_data_dtype(np.int16)
                    img_out = nib.Nifti1Image(gt_seg, img.affine, img.header)
                    nib.save(img_out, folder / 'T2.nii.gz')
                elif file.is_dir():
                    shutil.rmtree(folder / file.name)
                else:
                    os.remove(folder / file.name)

if __name__ == "__main__":
    re_name_data(Path('/localmount/volume-hd/users/uline/data_sets/new_cases'),verbose=True)