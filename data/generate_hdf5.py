from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import time
from typing import Dict
import pandas as pd
from collections import defaultdict

from preprocessing import interpolate_volume, map_size, rescale_image
from data_utils import get_thick_slices, filter_blank_slices_thick, sagittal_transform_coronal, sagittal_transform_axial

from utils.wm_logger import loggen

LOGGER =loggen('generate_hdf5')

class H5pyDataset:
    def __init__(self, params: Dict, preprocessing: bool = False):
        self.dataset_path = Path(params["dataset_path"])
        self.dataset_name = params["dataset_name"]
        self.preprocessing = preprocessing
        self.slice_thickness = params["thickness"]
        self.gt_name = params["gt_name"]
        self.volume_name = params["volume_name"]
        self.csv_file = Path(params["csv_file"])
        self.plane = params["plane"]
        self.datatype = params["datatype"]

        assert self.dataset_path.is_dir(), f"The provided paths are not valid: {self.dataset_path}!"

        dataframe = pd.read_csv(self.csv_file)
        dataframe = dataframe[dataframe["wmh_split"].str.contains(self.datatype)]

        self.subjects_dirs = [ self.dataset_path / row["imageid"] for index, row in dataframe.iterrows() ]

        self.data_set_size = len(dataframe)

    def _load_volumes(self, subject_path: Path):
        volume_img = nib.load(subject_path / self.volume_name)
        gt_img = nib.load(subject_path / self.gt_name)

        zooms, shape = volume_img.header.get_zooms(), volume_img.header.get_data_shape()

        volume_img = nib.as_closest_canonical(volume_img)
        gt_img = nib.as_closest_canonical(gt_img)

        if self.preprocessing:
            if zooms != (1.0, 1.0, 1.0):
                gt_img = interpolate_volume(gt_img,interpolation="nearest")
                volume_img = interpolate_volume(volume_img)
        gt_seg = np.asarray(
            gt_img.get_fdata(), dtype=np.int16
        )
        volume = np.asarray(
            volume_img.get_fdata(), dtype=np.uint8
        )
        if self.preprocessing:
            volume = map_size(volume,[256, 256, 256])
            gt_seg = map_size(gt_seg,[256, 256, 256])

            volume = rescale_image(volume)
            gt_seg = rescale_image(gt_seg)

        return volume, gt_seg, zooms

    def create_hdf5_dataset(self, blt: int):

        data_per_idx = defaultdict(lambda: defaultdict(list))

        for idx, current_subject in enumerate(self.subjects_dirs):
            try:
                start_d = time.time()

                LOGGER.info(
                    f"Volume Nr: {idx + 1} Processing MRI Data from {current_subject.name}/{self.volume_name}"
                )

                volume, gt_img, zooms = self._load_volumes(current_subject)
                size, _ , _ = volume.shape

                if self.plane == "axial":
                    volume = sagittal_transform_axial(volume)
                    gt_img = sagittal_transform_axial(gt_img)

                if self.plane == "coronal":
                    volume = sagittal_transform_coronal(volume)
                    gt_img = sagittal_transform_coronal(gt_img)

                volume_thick = get_thick_slices(volume, self.slice_thickness)

                filter_volume , filter_labels = filter_blank_slices_thick(volume_thick, gt_img, threshold=blt)

                num_batch = volume_thick.shape[0]

                data_per_idx[f"{size}"]["volume"].extend(filter_volume)
                data_per_idx[f"{size}"]["seg"].extend(filter_labels)
                data_per_idx[f"{size}"]["zoom"].extend((zooms,) * num_batch)
                data_per_idx[f"{size}"]["subject"].append(
                        current_subject.name.encode("ascii", "ignore")
                )
            except Exception as e:
                LOGGER.info(f"Volume {size} Failed Reading Data. Error: {e}")
                continue


        for key, data_dict in data_per_idx.items():
            data_per_idx[key]["orig"] = np.asarray(data_dict["volume"], dtype=np.uint8)
            data_per_idx[key]["seg"] = np.asarray(data_dict["seg"], dtype=np.uint8)

        with h5py.File(self.dataset_name, "w") as hf:
            dt = h5py.special_dtype(vlen=str)
            for key, data_dict in data_per_idx.items():
                group = hf.create_group(f"{key}")
                group.create_dataset("orig_dataset", data=data_dict["volume"])
                group.create_dataset("aseg_dataset", data=data_dict["seg"])
                group.create_dataset("zoom_dataset", data=data_dict["zoom"])
                group.create_dataset("subject", data=data_dict["subject"], dtype=dt)
        
        LOGGER.info(
            "Successfully written {} in {:.3f} seconds".format(
                self.dataset_name, time.time() - start_d
            )
        )

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser("Creating hdf5 for trainning and validation")

    parser.add_argument("--dataset_name",
    type=str,
    help="name of the file to save the hdf5 file")

    parser.add_argument("--dataset_path",
    type=str,
    help="Directory with images to load")

    parser.add_argument("--thickness",
    type=int,
    default = 3,
    help="Number of pre- and succeding slices (default: 3)"
    )

    parser.add_argument("--gt_name",
    type=str,
    default="lesion.nii.gz",
    help="Default name of the segementation images. default is (lesion.nii.gz)"
    )

    parser.add_argument("--volumen_name",
    type=str,
    default="FLAIR.nii.gz",
    help = "Default name of the original images. default is (FLAIR.nii.gz)"
    )

    parser.add_argument("--csv_file",
    type=str,
    help="CSV file listing the splitting of the volumes")

    parser.add_argument("--plane",
    type=str,
    choices=["axial","coronal", "sagital"],
    default="axial",
    help = "Which plane to put into file (axial (default), coronal or sagittal)"
    )

    parser.add_argument("--datatype",
    type=str,
    default="train",
    choices=["train","val"],
    help="Type of the dataset to use in the genration of the hdf5 file")

    parser.add_argument("--p","--preprocessing",
    action="store_true",
    default=False,
    help="Turn on to add the preprocessing to your dataset"
    )

    args = parser.parse_args()

    dataset_params = {
        "dataset_name": args.dataset_name,
        "dataset_path": args.dataset_path,
        "thickness": args.thickness,
        "gt_name": args.gt_name,
        "volume_name":args.volumen_name,
        "csv_file":args.csv_file,
        "plane":args.plane,
        "datatype":args.datatype
    }

dataset_generator = H5pyDataset(params=dataset_params)
dataset_generator.create_hdf5_dataset(10)




