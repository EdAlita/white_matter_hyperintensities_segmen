import argparse
from pathlib import Path
import shutil

def reorganize_patient_folders(mainfolder, outputfolder):
    mainfolder_path = Path(mainfolder)
    outputfolder_path = Path(outputfolder)

    # Create the output folder if it doesn't exist
    outputfolder_path.mkdir(parents=True, exist_ok=True)

    # Iterate over each patient folder
    for patient_folder in mainfolder_path.glob('patient*/'):
        if patient_folder.is_dir():
            raw_folder = patient_folder / 'raw'
            patient_output_folder = outputfolder_path / patient_folder.name
            patient_output_folder.mkdir(exist_ok=True)

            # Define file names
            flair_file = f'{patient_folder.name}_FLAIR.nii.gz'
            t1w_file = f'{patient_folder.name}_T1W.nii.gz'
            consensus_gt_file = f'{patient_folder.name}_consensus_gt.nii.gz'
            lesion_file = f'{patient_folder.name}_lesion.nii.gz'

            # Copy FLAIR and T1W from raw folder to output patient folder
            for file_name in [flair_file, t1w_file]:
                source_file = raw_folder / file_name
                target_file = patient_output_folder / file_name
                if source_file.exists():
                    shutil.copy2(source_file, target_file)

            # Copy and rename consensus_gt to lesion in output patient folder
            consensus_gt_path = patient_folder / consensus_gt_file
            lesion_path = patient_output_folder / lesion_file
            if consensus_gt_path.exists():
                shutil.copy2(consensus_gt_path, lesion_path)

            print(f"Processed {patient_folder.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reorganize patient folders into an output folder.")
    parser.add_argument("mainfolder", type=str, help="Path to the main folder containing patient folders.")
    parser.add_argument("outputfolder", type=str, help="Path to the output folder where reorganized data will be saved.")
    args = parser.parse_args()
    reorganize_patient_folders(args.mainfolder, args.outputfolder)
