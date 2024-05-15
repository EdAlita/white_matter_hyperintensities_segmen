from pathlib import Path
from tqdm.auto import tqdm
import SimpleITK as sitk
import nibabel as nib
from scipy.ndimage import label as ndimage_label
from scipy.ndimage import binary_fill_holes
import numpy as np


class create_2_D_nnUnet:
    def __init__(self,
                 out_path: Path,
                 npz_path: Path,
                 th: float = 0.6):
        self.out_path = out_path
        self.npz_path = npz_path
        self.th = th

    def create(self):
        for folder in tqdm(sorted(self.out_path.iterdir()), desc='Evaluating....'):
            probs = []
            sagittal = np.load(self.npz_path / 'sagital' / f'{folder.name}.npz')
            _tmp = np.moveaxis(sagittal['softmax'][1, :, :, :], [0, 1, 2], [2, 0, 1])
            probs.append(_tmp)

            coronal = np.load(self.npz_path / f'coronal' / f'{folder.name}.npz')
            _tmp2 = np.moveaxis(coronal['softmax'][1, :, :, :], [0, 1, 2], [1, 0, 2])
            probs.append(_tmp2)

            axial = np.load(self.npz_path / f'axial' / f'{folder.name}.npz')
            probs.append(axial['softmax'][1, :, :, :])

            img = sitk.ReadImage(folder / 'lesion.nii.gz')
            out = np.max(probs, axis=0)
            out = np.asarray((out > self.th) * 1.0, np.uint8)

            for i in range(out.shape[0]):
                labelled_mask, num_labels = ndimage_label(out[i,:,:],structure=np.ones([3,3]))
                # Let us now remove all the too small regions.
                minimum_cc_sum = 5
                refined_mask = out[i,:,:].copy()

                for label in range(num_labels):
                    if np.sum(refined_mask[labelled_mask == label]) < minimum_cc_sum:
                        refined_mask[labelled_mask == label] = 0

                    out[i,:,:] = refined_mask

                new_img = sitk.GetImageFromArray(out)
                new_img.CopyInformation(img)

                sitk.WriteImage(new_img, folder / f'seg_nnUNet_25D.nii.gz')

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser("Bio-Agregation")

    parser.add_argument("--out_path", type=str, help="name of the out to create bio-agregation")

    parser.add_argument("--npz_path", type=str, help="name of the out to create bio-agregation")

    args = parser.parse_args()

    model = create_2_D_nnUnet(out_path=Path(args.out_path), npz_path=Path(args.npz_path))

    model.create()