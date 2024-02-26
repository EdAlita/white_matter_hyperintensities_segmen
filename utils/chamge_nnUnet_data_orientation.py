from pathlib import Path
import SimpleITK as sitk
from transform import axial_transform_coronal, axial_transform_sagital
import argparse

def transform(
        root_path: Path,
        new_space: str = "coronal",
        verbose: bool = False,
        inverse: bool = False):
    data = []
    if not root_path.exists():
        raise Exception(f"The path '{root_path}' does not exists.")
    if verbose: print(f"Transforming to {new_space} and flag to inverse: {inverse} ")
    for folder in sorted(root_path.iterdir()):
        if folder.is_dir():
            if verbose: print(f'Processing {folder.name}')
            for file in folder.glob('*.nii.gz'):
                if verbose: print(f'Processing {file.name}')
                img = sitk.ReadImage(file)
                array = sitk.GetArrayFromImage(img)

                if new_space == "coronal":
                    array_out = axial_transform_coronal(array, inverse=inverse)
                else:
                    array_out = axial_transform_sagital(array, inverse=inverse)

                result_image = sitk.GetImageFromArray(array_out)
                result_image.CopyInformation(img)

                sitk.WriteImage(result_image, folder / file.name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='change image orientation from nnUNet data')
    parser.add_argument("root_path", type=str, help="Directory where the dataset is located.")
    parser.add_argument("new_space", type=str, help="The new space to transform", default="coronal", choices=["coronal", "sagital"])
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help="Turn on the verbose flag")  # on/off flag
    parser.add_argument('-i', '--inverse', action='store_true', default=False,
                        help="Inverse the transformation")

    args = parser.parse_args()

    transform(Path(args.root_path), new_space=args.new_space, verbose=args.verbose, inverse=args.inverse)