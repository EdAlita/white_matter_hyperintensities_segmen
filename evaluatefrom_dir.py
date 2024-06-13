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
            gt =(nib.as_closest_canonical(nib.load(folder / 'lesion.nii.gz'))).get_fdata()
            gt[gt != 0] = 1
            gt = np.asarray(gt,dtype=np.uint8)
            folder_result = {}
            tqdm.write(f'Evaluating {folder.name}')
            for file in sorted(folder.glob('*.nii.gz')):
                if 'lesion' not in file.name:
                    vol = np.asarray((nib.as_closest_canonical(nib.load(file))).get_fdata(), dtype=np.uint8)
                    values = get_values(gt,vol,measures=('Dice','VS','HD95','Recall','F1'),voxelspacing=(1.0,1.0,1.0))
                    projection_name = (file.name.removesuffix('.nii.gz')).removeprefix('seg_')
                    tqdm.write(f"{projection_name}: Dice: {values['Dice']:.4f}, VS: {values['VS']:.4f}, HD95: {values['HD95']:.4f}")
                    folder_result['Dice_'+projection_name] = values['Dice']
                    folder_result['VS_' + projection_name] = values['VS']
                    folder_result['HD95_' + projection_name] = values['HD95']
                    folder_result['Recall_' + projection_name] = values['Recall']
                    folder_result['F1_' + projection_name] = values['F1']

            results[folder.name] = folder_result

        headers = ['ID'] + list(next(iter(results.values())).keys())
        rows = [
            [id] + list(values.values())
            for id, values in results.items()
        ]

        file_path = self.path.parent / f'{self.path.name}_result_overall.csv'
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows(rows)

    def create_model(self, prob:float):
        _out = np.zeros([256,256,256])
        for folder in sorted(self.path.iterdir()):
            axial = (nib.as_closest_canonical(nib.load(folder / 'segCNN_axial.nii.gz'))).get_fdata()
            sagittal = (nib.as_closest_canonical(nib.load(folder / 'segCNN_sagital.nii.gz'))).get_fdata()
            coronal = (nib.as_closest_canonical(nib.load(folder / 'segCNN_coronal.nii.gz'))).get_fdata()
            _out = (axial + coronal + sagittal)/3
            binary_model = np.asarray((_out > prob) * 1.0,np.uint8)
            test = nib.load(folder / 'segCNN_axial.nii.gz')
            nib.save(nib.Nifti1Image(binary_model, test.affine, test.header), folder / 'seg_CNN_model.nii.gz')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("Evaluating a trained model")

    parser.add_argument("--eval_path", type=str, help="name of the out to create bio-agregation")

    args = parser.parse_args()

    evaluate = EvaluateFromDir(root_dir=Path(args.eval_path))
    evaluate.evaluate()
