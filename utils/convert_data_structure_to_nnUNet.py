from pathlib import Path
import shutil
import argparse

def convert_data_structure_to_nnUNet(
    original_path:Path,
    destination_path:Path, 
    task_name:str, 
    data_type:str):
    """converts the data struture to nnUNet format

    Args:
        original_path (Path): Original path of the dataset
        destination_path (Path): Destination of the new dataset
        task_name (str): Name of the task to use on the nnUnet format
        data_type (str): Use of the FLAIR or combination of the FLAIR+T1
    """

    if not data_type in ['FLAIR','FLAIR+T1']:
        raise Exception(f'Datatype not recognize')

    print(f'Creating the nnUNet dataset for {data_type}')

    if not original_path.exists():
        raise Exception(f"The path '{original_path}' does not exist.")

    if not destination_path.exists():
        raise Exception(f"The path '{destination_path}' does not exist.")

    print(' ')
    print('Using directories: ')
    print(f' - origin path: {str(original_path)}')
    print(f' - destination path: {str(destination_path)}')
    
    # Check if task directory exists in the destination path
    task_path = destination_path / task_name

    if task_path.exists():
        print('New data structure already exists')
        return

    print(' ')
    print(f'Creating data structure for: {task_name}')

    # Create new directory structure
    (task_path / 'imagesTr').mkdir(parents=True, exist_ok=True)
    (task_path / 'imagesTs').mkdir(parents=True, exist_ok=True)
    (task_path / 'labelsTr').mkdir(parents=True, exist_ok=True)
    (task_path / 'labelsTs').mkdir(parents=True, exist_ok=True)

    # Function to handle the file copying and renaming
    def handle_files(source_folder,is_test=False,data_type='FLAIR'):
        if data_type =='FLAIR+T1':
            for folder in sorted(source_folder.iterdir()):
                if folder.is_dir():
                    for file in folder.glob('*.nii.gz'):
                        if 'lesion' in file.name:
                            # For segmentation files (labels)
                            dest_file = task_path / 'labelsTr' / (folder.name + '.nii.gz')
                            shutil.copy2(file, dest_file)
                        elif 'FLAIR' in file.name:
                            # For image files
                            suffix = '_0000.nii.gz'
                            dest_folder = 'imagesTs' if is_test else 'imagesTr'
                            dest_file = task_path / dest_folder / (folder.name + suffix)
                            shutil.copy2(file, dest_file)
                        elif 'T1' in file.name:
                            suffix = '_0001.nii.gz'
                            dest_folder = 'imagesTs' if is_test else 'imagesTr'
                            dest_file = task_path / dest_folder / (folder.name + suffix)
                            shutil.copy2(file, dest_file)
                        
        else:
            for folder in sorted(source_folder.iterdir()):
                if folder.is_dir():
                    for file in folder.glob('*.nii.gz'):
                        if 'lesion' in file.name:
                            # For segmentation files (labels)
                            dest_folder = 'labelsTs' if is_test else 'labelsTr'
                            dest_file = task_path / dest_folder / (folder.name + '.nii.gz')
                            shutil.copy2(file, dest_file)
                        elif 'FLAIR' in file.name:
                            # For image files
                            suffix = '_0000.nii.gz'
                            dest_folder = 'imagesTs' if is_test else 'imagesTr'
                            dest_file = task_path / dest_folder / (folder.name + suffix)
                            shutil.copy2(file, dest_file)

    # Process each set
    handle_files(original_path / 'train',is_test=False,data_type=data_type)
    #handle_files(original_path / 'Validation_Set')
    handle_files(original_path / 'test', is_test=True,data_type=data_type)

# Usage Example (Uncomment and modify the paths as needed)
if __name__ == "__main__":
     parser = argparse.ArgumentParser(description="Convert data structure for medical imaging")
     parser.add_argument("original_path", type=str, help="Path to the original data")
     parser.add_argument("destination_path", type=str, help="Path to the destination")
     parser.add_argument("task_name", type=str, help="Name of the task")
     parser.add_argument("data_type", type=str,choices=['FLAIR','FLAIR+T1'],help="data type structure to create")
     args = parser.parse_args()
     convert_data_structure_to_nnUNet(Path(args.original_path),Path(args.destination_path), args.task_name, args.data_type)
