from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import time
from typing import Dict
import pandas as pd
from collections import defaultdict

from preprocessing import interpolate_volume, map_size, rescale_image
from data_utils import get_thick_slices, filter_blank_slices_thick, sagittal_transform_coronal, sagittal_transform_axial, create_weight_mask

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
        self.volume2_name = params["volume2_name"]
        self.volume3_name = params["volume3_name"]
        self.csv_file = Path(params["csv_file"])
        self.plane = params["plane"]
        self.datatype = params["datatype"]
        self.max_weight= params["max_weight"]
        self.edge_weight = params["edge_weight"]
        self.gradient = params["gradient"]

        assert self.dataset_path.is_dir(), f"The provided paths are not valid: {self.dataset_path}!"

        dataframe = pd.read_csv(self.csv_file)
        dataframe = dataframe[dataframe["wmh_split"].str.contains(self.datatype)]

        self.subjects_dirs = [ self.dataset_path / row["imageid"] for index, row in dataframe.iterrows() ]

        self.data_set_size = len(dataframe)

    def _load_volumes(self, subject_path: Path):
        LOGGER.info(
            "Processing intensity image {}, image2 {}, image3 {} and ground truth segmentation {}".format(
                self.volume_name,self.volume2_name,self.volume3_name, self.gt_name
            )
        )
        volume_img = nib.load(subject_path / self.volume_name)
        gt_img = nib.load(subject_path / self.gt_name)
        if self.volume2_name is None:
            volume2_img = nib.load(subject_path / self.volume_name)
        else:
            volume2_img = nib.load(subject_path / self.volume2_name)
        if self.volume3_name is None:
            volume3_img = nib.load(subject_path / self.volume_name)
        else:
            volume3_img = nib.load(subject_path / self.volume3_name)

        zooms, shape = volume_img.header.get_zooms(), volume_img.header.get_data_shape()

        volume_img = nib.as_closest_canonical(volume_img)
        volume2_img = nib.as_closest_canonical(volume2_img)
        volume3_img = nib.as_closest_canonical(volume3_img)
        gt_img = nib.as_closest_canonical(gt_img)

        if self.preprocessing:
            if zooms != (1.0, 1.0, 1.0):
                gt_img = interpolate_volume(gt_img,interpolation="nearest")
                volume_img = interpolate_volume(volume_img)
        gt_seg = np.asarray(
            gt_img.get_fdata(), dtype=np.uint8
        )
        volume = np.asarray(
            volume_img.get_fdata(), dtype=np.uint8
        )
        volume2 = np.asarray(
            volume2_img.get_fdata(), dtype=np.uint8
        )
        volume3 = np.asarray(
            volume3_img.get_fdata(), dtype=np.uint8
        )
        if self.preprocessing:
            volume = map_size(volume,[256, 256, 256])
            gt_seg = map_size(gt_seg,[256, 256, 256])

            volume = rescale_image(volume)
            gt_seg = rescale_image(gt_seg)

        return volume, volume2, volume3, gt_seg, zooms

    def create_hdf5_dataset(self, blt: int):

        data_per_idx = defaultdict(lambda: defaultdict(list))

        for idx, current_subject in enumerate(self.subjects_dirs):
            try:
                start_d = time.time()

                LOGGER.info(
                    f"Volume Nr: {idx + 1} Processing MRI Data from {current_subject.name}/{self.volume_name}"
                )

                volume, volume2, volume3, gt_img, zooms = self._load_volumes(current_subject)
                size, _, _ = volume.shape

                if self.plane == "axial":
                    volume = sagittal_transform_axial(volume)
                    volume2 = sagittal_transform_axial(volume2)
                    volume3 = sagittal_transform_axial(volume3)
                    gt_img = sagittal_transform_axial(gt_img)

                if self.plane == "coronal":
                    volume = sagittal_transform_coronal(volume)
                    volume2 = sagittal_transform_axial(volume2)
                    volume3 = sagittal_transform_axial(volume3)
                    gt_img = sagittal_transform_coronal(gt_img)

                weight = create_weight_mask(
                    gt_img,
                    max_weight=self.max_weight,
                    max_edge_weight= self.edge_weight,
                    gradient= self.gradient
                )
                LOGGER.info(
                    "Created weights with max_w {}, gradient {},"
                    " edge_w {}".format(
                        self.max_weight,
                        self.gradient,
                        self.edge_weight))

                volume_thick = get_thick_slices(volume, self.slice_thickness)
                volume2_thick = get_thick_slices(volume2, self.slice_thickness)
                volume3_thick = get_thick_slices(volume3, self.slice_thickness)

                filter_volume, filter_volume2,filter_volume3, filter_labels, filter_weight = filter_blank_slices_thick(
                    volume_thick,volume2_thick,volume3_thick, gt_img, weight, threshold=blt)

                num_batch = volume_thick.shape[0]

                data_per_idx[f"{size}"]["volume"].extend(filter_volume)
                data_per_idx[f"{size}"]["volume2"].extend(filter_volume2)
                data_per_idx[f"{size}"]["volume3"].extend(filter_volume3)
                data_per_idx[f"{size}"]["seg"].extend(filter_labels)
                data_per_idx[f"{size}"]["weight"].extend(filter_weight)
                data_per_idx[f"{size}"]["zoom"].extend((zooms,) * num_batch)
                data_per_idx[f"{size}"]["subject"].append(
                        current_subject.name.encode("ascii", "ignore")
                )
            except Exception as e:
                LOGGER.info(f"Volume {size} Failed Reading Data. Error: {e}")
                continue

        for key, data_dict in data_per_idx.items():
            data_per_idx[key]["orig"] = np.asarray(data_dict["volume"], dtype=np.uint8)
            data_per_idx[key]["orig2"] = np.asarray(data_dict["volume2"], dtype=np.uint8)
            data_per_idx[key]["orig3"] = np.asarray(data_dict["volume3"], dtype=np.uint8)
            data_per_idx[key]["seg"] = np.asarray(data_dict["seg"], dtype=np.uint8)
            data_per_idx[key]["weight"] = np.asarray(data_dict["weight"], dtype=float)
        
        with h5py.File(self.dataset_name, "w") as hf:
            dt = h5py.special_dtype(vlen=str)
            for key, data_dict in data_per_idx.items():
                group = hf.create_group(f"{key}")
                group.create_dataset("orig_dataset", data=data_dict["volume"])
                group.create_dataset("orig2_dataset", data=data_dict["volume2"])
                group.create_dataset("orig3_dataset", data=data_dict["volume3"])
                group.create_dataset("aseg_dataset", data=data_dict["seg"])
                group.create_dataset("weight_dataset", data=data_dict["weight"])
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
    parser.add_argument("--volumen2_name",
    type=str,
    default=None,
    help ="Default name of the original images. default is (None)"
    )
    parser.add_argument("--volumen3_name",
    type=str,
    default=None,
    help ="Default name of the original images. default is (None)"
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

    parser.add_argument(
        "--max_w",
        type=int,
        default=5,
        help="Overall max weight for any voxel in weight mask. Default=5",
    )

    parser.add_argument(
        "--edge_w",
        type=int,
        default=5,
        help="Weight for edges in weight mask. Default=5",
    )

    parser.add_argument(
        "--no_grad",
        action="store_true",
        default=False,
        help="Turn on to only use median weight frequency (no gradient)",
    )

    args = parser.parse_args()

    dataset_params = {
        "dataset_name": args.dataset_name,
        "dataset_path": args.dataset_path,
        "thickness": args.thickness,
        "gt_name": args.gt_name,
        "volume_name":args.volumen_name,
        "volume2_name": args.volumen2_name,
        "volume3_name": args.volumen3_name,
        "csv_file":args.csv_file,
        "plane":args.plane,
        "datatype":args.datatype,
        "max_weight": args.max_w,
        "edge_weight": args.edge_w,
        "gradient": not args.no_grad
    }

    dataset_generator = H5pyDataset(params=dataset_params)
    dataset_generator.create_hdf5_dataset(15)




