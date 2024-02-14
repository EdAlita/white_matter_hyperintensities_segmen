from pathlib import Path
import shutil
import argparse
import nibabel as nib
import csv


def get_mri_information(
    data_path:Path,
    csv_data:Path):
    data = []
    if not data_path.exists():
        raise Exception(f"The path '{data_path}' does not exists.")
    
    for folder in sorted(data_path.iterdir()):
        if folder.is_dir():
            folder_name = folder.name
            for file in folder.glob('*.nii.gz'):
                for element in ['T1','T2','FLAIR','lesion','wmh']:
                    if element in file.stem:
                        file_type = file.stem
                        image = nib.load(file)
                        header = image.header
                        size = header.get_data_shape()
                        voxels = header.get_zooms()
                        row = {"dataset": folder_name,
                                    "id": file_type,
                                    "image size":size,
                                    "voxels size":voxels}
                        data.append(row)
                        print(f'Analyzing folder: {folder_name}: {file_type}')
    
    keys = data[0].keys()
    with open(csv_data,"a") as file:
        writer_obj = csv.DictWriter(file,keys)
        writer_obj.writerows(data)
        file.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creates csv of information from the datasets provide.")
    parser.add_argument("data_path",type=str, help="the path to the data_set to analyze.")
    parser.add_argument("csv_data", type=str, help="Path for the csv to save the info retrive from the dataset")

    args = parser.parse_args()
    get_mri_information(Path(args.data_path),Path(args.csv_data))









