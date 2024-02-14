from pathlib import Path
import csv
import shutil
import argparse

def copyfiles(root:Path,data_type:str,dest:Path):
    if not root.exists():
        raise Exception(f"The path '{root}' does not exists.")
    if not dest.exists():
        raise Exception(f"The path '{dest}' does not exists.")

    if not data_type in ['test','train','val']:
        raise Exception(f"The data_type '{data_type}' not supported")

    import csv

    with open(root / "wmh_overall.csv", "r") as f:
        reader = csv.reader(f, delimiter=",")
        for i, line in enumerate(reader):
            if not i == 0: 
                if line[1] == data_type:
                   print(f"Copying '{root / line[0]}'")
                   shutil.copytree(root / line[0], dest  / line[0], dirs_exist_ok = True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy files for new split from csv file.")
    parser.add_argument("root",type=str, help="the path to the data_set to analyze.")
    parser.add_argument("data_type", type=str,choices=['test','train','val'], help="data type to analyze")
    parser.add_argument("dest",type=str, help="destination path of the dataset to copy.")

    args = parser.parse_args()
    copyfiles(Path(args.root),args.data_type,Path(args.dest))
