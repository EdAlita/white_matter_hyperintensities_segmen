# Run Inference

## Excecution

To execute the inference, provide the necessary paths and configuration settings through command line arguments:

- `--dataset_path`: Location of the dataset to load for inference.
- `--volume_name`: Name of the volume to load for inference.
- `--volume2_name`: Name of the second volume to load for inference.
- `--volume3_name`: Name of the third volume to load for inference.
- `--cfg_file`: Path to the configuration file.
- `--csv_file`: Path for the csv file describing the dataset.
- `--plane`: Name of the plane to load in inference.
- `--out_path`: Outpath to save the volumes.
- `--ckpt_path` : Path to ckpt to load.
- `--num_channels`: Number of input channels for the model [1,2,3]

Example of axial inference with 2 inputs: 

```bash
python3 inference.py --dataset_path path/to/dataset \
--volume_name FLAIR.nii.gz \
--volume2_name T2.nii.gz \
--cfg_path path/to/config.yaml \
--csv_file path/to/dataset/wmh_overall.csv \
--plane axial \
--ckpt_path path/to/model/checkpoint/ \
--num_channels 2 \
```
