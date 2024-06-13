# Run trainning

The trainning creates a folder strcuture where you run the script

## Run with hdf5 file

### Execution

For this you need to run the HDF5 generator file from data folder

To execute the training, provide the necessary paths and configuration settings through command line arguments:
``
- `--config_path`: Path to the configuration file.
- `--train_path`: Path to the hdf5 file of train data.
- `--val_path`: Path to the hdf5 file of validation data.
- `--num_channels`: Number of input channels for the model [1,2,3]

Example: 

```bash
python train.py --config_path path/to/config.yaml \
--train_path path/to/train.hdf5 \
--val_path path/to/val.hdf5 \
--num_channels 2
```

## Run without hdf5 file

### Execution

To execute the training, provide the necessary paths and configuration settings through command line arguments:
``
- `--config_path`: Path to the configuration file.
- `--train_path`: Path to the training dataset.
- `--val_path`: Path to the validation dataset.
- `--num_channels`: Number of input channels for the model. [1,2,3]

Example: 

```bash
python train-UKbiobank.py --config_path path/to/config.yaml \
--train_path path/to/train/dataset \
--val_path path/to/val/dataset \
--num_channels 2
```
