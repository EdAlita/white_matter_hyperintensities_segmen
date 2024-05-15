from typing import Dict
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
import shutil
import os

class file_struct:
    def __init__(self,
                 params: Dict):
        self.root_path = Path(params['root_path'])
        self.datatype = params['datatype']
        self.csv_path = self.root_path / 'wmh_overall.csv'
        self.out_path = Path(params['out_path'])

        assert self.root_path.is_dir(), f"The provided paths are not valid: {self.root_path}!"

        dataframe = pd.read_csv(self.csv_path)
        self.dataframe = dataframe[dataframe["wmh_split"].str.contains(self.datatype)]['imageid'].tolist()

    def create_file_struct(self):
        for folder in tqdm(self.dataframe, desc='Creating....'):

            os.makedirs(self.out_path / folder, exist_ok=True)

            shutil.copy(self.root_path / folder / 'lesion.nii.gz', self.out_path / folder)


if __name__ == '__main__':
    import argparse

    import argparse

    parser = argparse.ArgumentParser("create file struct")

    parser.add_argument("--root_path", type=str, help="folder from the original dataset")

    parser.add_argument("--csv_name", type=str, help="name of csv for the detail of the dataset")

    parser.add_argument("--out_path", type=str, help="folder to create the new struct")

    parser.add_argument("--datatype",
                        type=str,
                        default="test",
                        choices=["test", "val"],
                        help="Type of the dataset to use")

    args = parser.parse_args()

    dataset_params = {
        "root_path": args.root_path,
        "csv_name": args.csv_name,
        "out_path": args.out_path,
        "datatype": args.datatype,
    }

    filestruct_generator = file_struct(params=dataset_params)

    filestruct_generator.create_file_struct()
