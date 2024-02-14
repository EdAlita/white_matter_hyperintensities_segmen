import os

if __name__ == "__main__":
    base_dir = os.getcwd()
    nnUNet_raw = os.path.join(base_dir,'nnUNet_raw')
    nnUNet_preprocessed = os.path.join(base_dir,'nnUNet_preprocessed')
    results_folder = os.path.join(base_dir,'nnUNet_results')

    print(nnUNet_raw)

    os.environ["nnUNet_raw_data_base"] = str(nnUNet_raw)
    os.environ["nnUNet_preprocessed"] = str(nnUNet_preprocessed)
    os.environ["RESULTS_FOLDER"] = str(results_folder)