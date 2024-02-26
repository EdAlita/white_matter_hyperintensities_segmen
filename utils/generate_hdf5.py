from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import time
from typing import Dict
import pandas as pd
from collections import defaultdict

from data.preprocessing import interpolate_volume, map_size, rescale_image
from data.data_utils import get_thick_slices, filter_blank_slices_thick, sagittal_transform_coronal, sagittal_transform_axial

class H5pyDataset:
    def __init__(self, params: Dict, preprocessing: bool = False):
        self.dataset_path = Path(params['dataset_path'])
        self.dataset_name = params['dataset_name']
        self.preprocessing = preprocessing
        self.slice_thickness = params["thickness"]
        self.gt_name = params["gt_name"]
        self.volume_name = params['volume_name']
        self.csv_file = Path(params["csv_file"])
        self.plane = params["plane"]
        self.datatype = params["datatype"]

        assert self.dataset_path.is_dir(), f'The provided paths are not valid: {self.dataset_path}!'

        dataframe = pd.read_csv(self.csv_file)
        dataframe = dataframe[dataframe['wmh_split'].str.contains(self.datatype)]

        self.subjects_dirs = [ self.dataset_path / row['imageid'] for index, row in dataframe.iterrows() ]

        self.data_set_size = len(dataframe)

    def _load_volumes(self, subject_path: Path):
        volume_img = nib.load(subject_path / self.volume_name)
        gt_img = nib.load(subject_path / self.gt_name)

        zooms, shape = volume_img.header.get_zooms(), volume_img.header.get_data_shape()

        volume_img = nib.as_closest_canonical(volume_img)
        gt_img = nib.as_closest_canonical(gt_img)

        if self.preprocessing:
            if zooms != (1.0, 1.0, 1.0):
                gt_img = interpolate_volume(gt_img,interpolation='nearest')
                volume_img = interpolate_volume(volume_img)
        gt_seg = np.asarray(
            gt_img.get_fdata()   , dtype=np.int16
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

        data_per_size = defaultdict(lambda: defaultdict(list))

        for idx, current_subject in enumerate(self.subjects_dirs):

            volume, gt_img, zooms = self._load_volumes(current_subject)
            size, _ , _ = volume.shape

            if self.plane == 'axial':
                volume = sagittal_transform_axial(volume)
                gt_img = sagittal_transform_axial(gt_img)

            if self.plane == 'coronal':
                volume = sagittal_transform_coronal(volume)
                gt_img = sagittal_transform_coronal(gt_img)

            volume_thick = get_thick_slices(volume, self.slice_thickness)

                #volume, gt_img = filter_blank_slices_thick(
                 #   volume_thick, gt_img, threshold=blt)

            num_batch = volume_thick.shape[0]

            data_per_size[f"{size}"]["volume"].extend(volume_thick)
            data_per_size[f"{size}"]["seg"].extend(gt_img)
            data_per_size[f"{size}"]["zoom"].extend((zooms,) * num_batch)
            data_per_size[f"{size}"]["subject"].append(
                    current_subject.name.encode("ascii", "ignore")
            )


        for key, data_dict in data_per_size.items():
            data_per_size[key]["orig"] = np.asarray(data_dict["orig"], dtype=np.uint8)
            data_per_size[key]["aseg"] = np.asarray(data_dict["aseg"], dtype=np.uint8)

        with h5py.File(self.dataset_name, "w") as hf:
            dt = h5py.special_dtype(vlen=str)
            for key, data_dict in data_per_size.items():
                group = hf.create_group(f"{key}")
                group.create_dataset("orig_dataset", data=data_dict["orig"])
                group.create_dataset("aseg_dataset", data=data_dict["aseg"])
                group.create_dataset("zoom_dataset", data=data_dict["zoom"])
                group.create_dataset("subject", data=data_dict["subject"], dtype=dt)

if __name__ == "__main__":
    dataset_params = {
        "dataset_name": "/localmount/volume-hd/users/uline/fastCNN_split/axial/training_set_axial.hdf5",
        "dataset_path": "/localmount/volume-hd/users/uline/data_sets/CVD/",
        "thickness" : 3,
        "gt_name": "lesion.nii.gz",
        "volume_name":"FLAIR.nii.gz",
        "csv_file":"/localmount/volume-hd/users/uline/data_sets/CVD/wmh_overall.csv",
        "plane":"axial",
        "datatype":"train"
    }

dataset_generator = H5pyDataset(params=dataset_params, preprocessing=False)
dataset_generator.create_hdf5_dataset(50)




