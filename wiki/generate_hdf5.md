# Run Generate HDF5

## Prepare data csv

In order to sort the data into the split of val, test and train you need to create a csv with this structure:

| imageid | wmh_split |
| :---:   | :---: |
| 69c2dddd-cd4a-4501-887a-f7d6899d91e3 | train   |
| 790f3bed-e979-47d3-8c52-6ae84eb9a0ab | test   |
| 18978ce2-7faa-46e6-9ab6-36e931712365 | val   |

Use the same imageid as the folder structure as follow below

```bash
Dataset
├── 69c2dddd-cd4a-4501-887a-f7d6899d91e3
│   ├── FLAIR.nii.gz
│   ├── T1.nii.gz
│   ├── T2.nii.gz
│   └── lesion.nii.gz
├── 790f3bed-e979-47d3-8c52-6ae84eb9a0ab
│   ├── FLAIR.nii.gz
│   ├── T1.nii.gz
│   ├── T2.nii.gz
│   └── lesion.nii.gz
├── 18978ce2-7faa-46e6-9ab6-36e931712365
│   ├── FLAIR.nii.gz
│   ├── T1.nii.gz
│   ├── T2.nii.gz
│   └── lesion.nii.gz
└── wmh_overall.csv
```

## Excecution

To execute the inference, provide the necessary paths and configuration settings through command line arguments:

- `--dataset_name`: name of the file to save the hdf5 file.
- `--dataset_path`: directory with images to load.
- `--thickness` : Number of pre- and succeding slices (default: 3)
- `--gt_name` : Default name of the segementation images. Default is (lesion.nii.gz)
- `--volume_name` : Default name of the original images. default is (FLAIR.nii.gz)
- `--volume2_name` : Default name of the original images. default is (None)
- `--volume3_name` : Default name of the original images. default is (None)
- `--csv_file` : CSV file listing the splitting of the volumes
- `--plane` : Which plane to put into file (axial (default), coronal or sagittal)
- `--datatype` : Type of the dataset to use in the genration of the hdf5 file (train or val)
- `--max_w` : Overall max weight for any voxel in weight mask. Default=5
- `--edge_w` : Weight for edges in weight mask. Default=5
- `--no_grad` : Turn on to only use median weight frequency (no gradient)

Example of the use of generate_hdf5

```bash
python3 data/generate_hdf5.py --dataset_name train_axial_FlairT1.hdf5 \
--dataset_path /localmount/volume-hd/users/uline/data_sets/CVD \
--csv_file  /localmount/volume-hd/users/uline/data_sets/CVD/wmh_overall.csv \
--thickness 3 \
--gt_name lesion.nii.gz \
--volumen2_name FLAIR.nii.gz \
--volumen2_name T1.nii.gz \
--plane axial \
--datatype train
```

