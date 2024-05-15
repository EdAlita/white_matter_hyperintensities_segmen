import json
import argparse
from pathlib import Path

def create_dataset_json(parent_dir, output_file, data_type="FLAIR+T1"):
    """
    Creates a JSON file that describes the dataset structure for medical imaging data.

    Args:
    parent_dir (str): The directory where the dataset is located.
    output_file (str): The path where the JSON file will be saved.
    data_type (str): The type of medical imaging data ('FLAIR' or 'FLAIR+T1'). Default is 'FLAIR'.
    """
    output_path = Path(output_file)
    # Check if the output file already exists
    if output_path.exists():
        print(f"File '{output_file}' already exists. No action taken.")
        return

    # Define the structure of the JSON file
    dataset_json = {
        "modality": {"0": "FLAIR"},
        "labels": {
            "0": "background",
            "1": "WM_hyper"
        },
        "numTraining": 0,
        "numTest": 0,
        "training": [],
        "test": []
    }

    # Paths for training and test data
    parent_path = Path(parent_dir)
    training_images_path = parent_path / "imagesTr"
    training_labels_path = parent_path / "labelsTr"
    test_images_path = parent_path / "imagesTs"
    test_labels_path = parent_path / "labelsTs"

    if data_type == "FLAIR+T1":
        dataset_json["modality"] = {"0": "FLAIR", "1": "T1"}
    
    # Scan for training images and labels
    if training_images_path.exists() and training_labels_path.exists():
        training_images = sorted([f.name for f in training_images_path.glob('*.nii.gz')])
        training_labels = sorted([f.name for f in training_labels_path.glob('*.nii.gz')])
        for img in training_labels:
            dataset_json["training"].append({
                    "image": str(training_images_path / img),
                    "label": str(training_labels_path / img)
                    })
            
        dataset_json["numTraining"] = len(dataset_json["training"])

    # Scan for test images
    if test_images_path.exists():
        test_images = sorted([f.name for f in test_labels_path.glob('*.nii.gz')])
        for img in test_images:
            dataset_json["test"].append(str(test_images_path / img))

        dataset_json["numTest"] = len(dataset_json["test"])

    # Write to JSON file
    with output_path.open('w') as outfile:
        json.dump(dataset_json, outfile, indent=4)

    print(f"Dataset JSON created at {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a dataset JSON file for medical imaging data.")
    parser.add_argument("parent_dir", type=str, help="Directory where the dataset is located.")
    parser.add_argument("output_file", type=str, help="Path where the JSON file will be saved.")
    parser.add_argument("--data_type", type=str, choices=["FLAIR", "FLAIR+T1"], default="FLAIR+T1", help="Type of medical imaging data. Default is 'FLAIR'.")

    args = parser.parse_args()
    create_dataset_json(args.parent_dir, args.output_file, args.data_type)
