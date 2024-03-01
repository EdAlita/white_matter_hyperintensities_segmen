from torch.utils.data import DataLoader
import yacs.config

from data import dataset as dset
from utils.wm_logger import loggen
from config.defaults import get_cfg_defaults

logger =loggen(__name__)


def get_dataloader(cfg: yacs.config.CfgNode, mode: str):
    assert mode in ['train', 'val'], f'dataloader mode is invalid: {mode}'

    if mode == 'train':
        data_path = cfg.DATA.PATH_HDF5_TRAIN
        shuffle = True
        logger.info(f"loading {mode.capitalize()} data ... from {data_path}")
        dataset = dset.Dataset_V1(dataset_path=data_path,cfg=cfg)
    else:
        data_path = cfg.DATA.PATH_HDF5_VAL
        shuffle = False
        logger.info(f"loading {mode.capitalize()} data ... from {data_path}")
        dataset = dset.Dataset_V1(dataset_path=data_path, cfg=cfg)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        shuffle=shuffle,
        pin_memory=True
    )
    return dataloader

if __name__ == '__main__':
    cfg = get_cfg_defaults()
    cfg.merge_from_file('/home/uline/Desktop/white_matter_hyperintensities_segmen/config/FastSurferCNN_axial.yaml')

    train_loader = get_dataloader(cfg=cfg, mode='train')
    val_loader = get_dataloader(cfg=cfg, mode='val')