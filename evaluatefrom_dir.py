from pathlib import Path
import nibabel as nib
from metrics.eval_metrics import get_values
import csv
import numpy as np
from tqdm import tqdm


class EvaluateFromDir:
    def __init__(self,
                 root_dir: Path):
        self.path = root_dir

    def evaluate(self):
        results = {}

        for folder in tqdm(sorted(self.path.iterdir()),desc='Evaluating....'):
            gt = (nib.as_closest_canonical(nib.load(folder / 'lesion.nii.gz'))).get_fdata()
            folder_result = {}
            for file in sorted(folder.glob('*.nii.gz')):
                if 'lesion' not in file.name:
                    vol = (nib.as_closest_canonical(nib.load(file))).get_fdata()
                    values = get_values(gt,vol,measures=('Dice','VS','HD95'),voxelspacing=(1.0,1.0,1.0))
                    projection_name = (file.name.removesuffix('.nii.gz')).removeprefix('seg_')
                    folder_result['Dice_'+projection_name] = values['Dice']
                    folder_result['VS_' + projection_name] = values['VS']
                    folder_result['HD95_' + projection_name] = values['HD95']

            results[folder.name] = folder_result

        headers = ['ID'] + list(next(iter(results.values())).keys())
        rows = [
            [id] + list(values.values())
            for id, values in results.items()
        ]

        file_path = self.path / 'result_overall.csv'
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows(rows)

    def create_model(self, prob:float):
        for folder in sorted(self.path.iterdir()):
            sum = 0
            for file in sorted(folder.glob('*.nii.gz')):
                affine = nib.load(file).affine
                header = nib.load(file).header
                if 'lesion' not in file.name:
                    sum += (nib.as_closest_canonical(nib.load(file))).get_fdata()
            out = sum/3
            binary_model = (out > prob).astype(np.uint8)
            nib.save(nib.Nifti1Image(binary_model, affine, header ), folder / 'seg_2.5_D.nii.gz')


if __name__ == '__main__':
    evaluate = EvaluateFromDir(root_dir=Path('/localmount/volume-hd/users/uline/segmentation_results/nn_Unet/CVD'))
    #evaluate.evaluate()
    evaluate.create_model(prob=0.3)