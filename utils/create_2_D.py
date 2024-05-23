from pathlib import Path
from tqdm.auto import tqdm
import SimpleITK as sitk
import nibabel as nib
from scipy.ndimage import label as ndimage_label
from scipy.ndimage import binary_fill_holes
import numpy as np


class create_2_D:
    def __init__(self,
                 out_path: Path,
                 npz_path: Path,
                 th: float = 0.9):
        self.out_path = out_path
        self.npz_path = npz_path
        self.th = th

    def create(self):
        for folder in tqdm(sorted(self.out_path.iterdir()), desc='Evaluating....'):
            probs = []
            for model in ['UNET','CNN']:
                sagittal = np.load(self.npz_path / f'{model}_sagital' / f'{folder.name}.npz')
                probs.append(sagittal['softmax'][:, 1, :, :])

                coronal = np.load(self.npz_path / f'{model}_coronal' / f'{folder.name}.npz')
                _tmp2 = np.moveaxis(coronal['softmax'][:, 1, :, :], [0, 1, 2], [1, 0, 2])
                probs.append(_tmp2)

                axial = np.load(self.npz_path / f'{model}_axial' / f'{folder.name}.npz')
                _tmp = np.moveaxis(axial['softmax'][:, 1, :, :], [0, 1, 2], [2, 0, 1])
                probs.append(_tmp)

                img = nib.load(folder / 'lesion.nii.gz')
                out = np.max(probs, axis=0)
                out = np.asarray((out > self.th) * 1.0, np.uint8)

                for i in range(out.shape[0]):
                    labelled_mask, num_labels = ndimage_label(out[i,:,:],structure=np.ones([3,3]))
                    label_size = [(labelled_mask == label).sum() for label in range(num_labels + 1)]
                    # Let us now remove all the too small regions.
                    minimum_cc_sum = 5
                    refined_mask = out[i,:,:].copy()
                    for label, size in enumerate(label_size):
                        if size < minimum_cc_sum:
                            refined_mask[labelled_mask == label] = 0

                    out[i, :, :] = refined_mask

                for i in range(out.shape[0]):
                    fill_mask = binary_fill_holes(out[i, :, :]).astype(int)
                    out[i, :, :] = fill_mask

                new_img = nib.Nifti1Image(out, img.affine, img.header)

                nib.save(new_img, folder / f'seg{model}_25D.nii.gz')



if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser("Bio-Agregation")

    parser.add_argument("--out_path", type=str, help="name of the out to create bio-agregation")

    parser.add_argument("--npz_path", type=str, help="name of the out to create bio-agregation")

    args = parser.parse_args()

    model = create_2_D(out_path=Path(args.out_path), npz_path=Path(args.npz_path))

    model.create()