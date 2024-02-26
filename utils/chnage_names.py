from pathlib import Path
import os


import shutil
def re_name_data(
        root_path: Path,
        verbose: bool = False
):
    if not root_path.is_dir(): raise NotADirectoryError()
    for folder in sorted(root_path.iterdir()):
        if folder.is_dir():
            for file in folder.iterdir():
                if 'lesion' in file.name or 'wmh' in file.name:
                    if verbose: print(folder / file.name, folder / 'lesion.nii.gz')
                    os.rename(folder / file.name, folder / 'lesion.nii.gz')
                elif 'FLAIR' in file.name:
                    if verbose: print(folder / file.name, folder / 'FLAIR.nii.gz')
                    os.rename(folder / file.name, folder / 'FLAIR.nii.gz')
                elif 'T1W' in file.name:
                    if verbose: print(folder / file.name, folder / 'T1W.nii.gz')
                    os.rename(folder / file.name, folder / 'T1W.nii.gz')
                elif file.is_dir():
                    shutil.rmtree(folder / file.name)
                else:
                    os.remove(folder / file.name)

if __name__ == "__main__":
    re_name_data(Path('/localmount/volume-hd/users/uline/data_sets/univ_Ljubljana'))