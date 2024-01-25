import SimpleITK as sitk
from pathlib import Path
import argparse

def convert_nhdr_to_nii_gz(folder_path):
    # Check if the folder exists
    if not folder_path.is_dir():
        print(f"The path {folder_path} does not exist or is not a directory.")
        return

    # Iterate over all files and subfolders in the folder
    for item in folder_path.iterdir():
        if item.is_file() and item.suffix == ".nhdr":
            # Read the NHDR file
            img = sitk.ReadImage(str(item))

            # Construct the new file name with .nii.gz extension
            new_file_path = item.with_suffix(".nii.gz")

            # Write the image in NII.GZ format
            sitk.WriteImage(img, str(new_file_path))
            print(f"Converted '{item.name}' to '{new_file_path.name}'")
        elif item.is_dir():
            # If the item is a directory, process it recursively
            convert_nhdr_to_nii_gz(item)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert NHDR files to NII.GZ format in a folder and its subfolders")
    parser.add_argument("folder_path", type=str, help="Path to the root folder containing NHDR files")

    args = parser.parse_args()
    folder_path = Path(args.folder_path)
    
    convert_nhdr_to_nii_gz(folder_path)
