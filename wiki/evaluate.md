# Run view aggregation and Evaluate script

## Excecution view aggration

To execute the view aggrgation, provide the necessary paths and configuration settings through command line arguments:

- `--out_path` : path of the place create the view aggregation models.
- `--npz_path` : path of the place with inference files to use.

Example to use: 

```bash
python3 utils/create_2_D.py --out_path /localmount/volume-hd/users/uline/segmentation_results/miccai_2017/transfer_lr/test_set \
 --npz_path /localmount/volume-hd/users/uline/segmentation_results/miccai_2017/transfer_lr/out
```

For the output file this structure needs to exists

```bash
Outfolder
├── 69c2dddd-cd4a-4501-887a-f7d6899d91e3
│   └── lesion.nii.gz
├── 790f3bed-e979-47d3-8c52-6ae84eb9a0ab
│   └── lesion.nii.gz
└──18978ce2-7faa-46e6-9ab6-36e931712365
    └── lesion.nii.gz
```
Also the npz folder file needs this structure

```bash
out_dir
├── UNET_axial
├── UNET_sagital
├── UNET_coronal
├── CNN_axial
├── CNN_sagital
└── CNN_coronal
```
## Excecution for evaluation script

To execute the view aggrgation, provide the necessary paths and configuration settings through command line arguments:

- `--eval_path` : path of the place create a evaluation path

Example to use: 

```bash
python3 evaluatefrom_dir.py --eval_path /localmount/volume-hd/users/uline/segmentation_results/miccai_2017/transfer_lr/test_set
```
