from torch.utils.data import DataLoader
import yacs.config

from data import dataset as dset
from utils.wm_logger import loggen
from config.defaults import get_cfg_defaults
import torchio as tio
from data.augmentation import ToTensor_2input, ToTensor_1input, ToTensor_3input

logger =loggen(__name__)


def get_dataloader(cfg: yacs.config.CfgNode, mode: str):
    assert mode in ['train', 'val'], f'dataloader mode is invalid: {mode}'

    if mode == 'train':
        data_path = cfg.DATA.PATH_HDF5_TRAIN
        shuffle = True
        logger.info(f"loading {mode.capitalize()} data ... from {data_path}")



        if cfg.MODEL.NUM_CHANNELS == 14:
            include = ["img","img2","label", "weight"]
            include_img = ["img","img2"]
        elif cfg.MODEL.NUM_CHANNELS == 21:
            include = ["img", "img2", "img3", "label", "weight"]
            include_img = ["img", "img2", "img3"]
        else:
            include = ["img","label", "weight"]
            include_img = ["img"]

        # Scales
        scaling = tio.RandomAffine(
            scales=(0.8, 1.15),
            degrees=0,
            translation=(0, 0, 0),
            isotropic=True,  # If True, scaling factor along all dimensions is the same
            center="image",
            default_pad_value="minimum",
            image_interpolation="linear",
            include=include,
        )

        #rotation
        rot = tio.RandomAffine(
            scales=(1.0, 1.0),
            degrees=10,
            translation=(0, 0, 0),
            isotropic= True,
            center="image",
            default_pad_value="minimum",
            image_interpolation="linear",
            include=include
        )

        #Angleflip
        angf = tio.RandomAffine(
            scales=(1.0, 1.0),
            degrees=180,
            translation=(0, 0, 0),
            isotropic= True,
            center="image",
            default_pad_value="minimum",
            image_interpolation="linear",
            include=include
        )
        # Translation
        tl = tio.RandomAffine(
            scales=(1.0, 1.0),
            degrees=0,
            translation=(15.0, 15.0, 0),
            isotropic=True,  # If True, scaling factor along all dimensions is the same
            center="image",
            default_pad_value="minimum",
            image_interpolation="linear",
            include=include,
        )

        # Bias Field
        bias_field = tio.transforms.RandomBiasField(
            coefficients=0.5, order=3, include=include_img
        )
        # Gaussian Noise
        random_noise_transform = tio.RandomNoise(
            mean=0,  # Default is 0, but you can adjust this if needed
            std=(0.01, 0.1),  # You can provide a range for standard deviation
            p=0.5,  # Probability of applying this transform
            include=include_img
        )

        all_augs = [
            rot,
            scaling,
            angf,
            tl,
            bias_field,
            random_noise_transform
        ]

        transform = tio.Compose(
            [tio.OneOf(all_augs, p=0.5)], include=include
        )

        dataset = dset.Dataset_t1(dataset_path=data_path,cfg=cfg,transforms=transform)
    else:
        data_path = cfg.DATA.PATH_HDF5_VAL
        shuffle = False
        logger.info(f"loading {mode.capitalize()} data ... from {data_path}")



        if cfg.MODEL.NUM_CHANNELS == 14:
            tensor_to_include = ToTensor_2input()
        elif cfg.MODEL.NUM_CHANNELS == 21:
            tensor_to_include = ToTensor_3input()
        else:
            tensor_to_include = ToTensor_1input()

        all_augs = [
            tensor_to_include,
        ]

        transform = tio.Compose(all_augs)

        dataset = dset.Dataset_V1(dataset_path=data_path, cfg=cfg, transforms=transform)



    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        shuffle=shuffle,
        pin_memory=True
    )
    return dataloader

def get_dataloader_biobank(cfg: yacs.config.CfgNode, mode: str, data_path=None, img_list=None):
    assert mode in ['train', 'val'], f'dataloader mode is invalid: {mode}'

    if mode == 'train':

        shuffle=True

        logger.info(f"loading {mode.capitalize()} data ... ")

        if cfg.MODEL.NUM_CHANNELS == 14:
            include = ["img","img2","label", "weight"]
            include_img = ["img","img2"]
        elif cfg.MODEL.NUM_CHANNELS == 21:
            include = ["img", "img2", "img3", "label", "weight"]
            include_img = ["img", "img2", "img3"]
        else:
            include = ["img","label", "weight"]
            include_img = ["img"]

        # Scales
        scaling = tio.RandomAffine(
            scales=(0.8, 1.15),
            degrees=0,
            translation=(0, 0, 0),
            isotropic=True,  # If True, scaling factor along all dimensions is the same
            center="image",
            default_pad_value="minimum",
            image_interpolation="linear",
            include=include,
        )

        #rotation
        rot = tio.RandomAffine(
            scales=(1.0, 1.0),
            degrees=10,
            translation=(0, 0, 0),
            isotropic= True,
            center="image",
            default_pad_value="minimum",
            image_interpolation="linear",
            include=include
        )

        #Angleflip
        angf = tio.RandomAffine(
            scales=(1.0, 1.0),
            degrees=180,
            translation=(0, 0, 0),
            isotropic= True,
            center="image",
            default_pad_value="minimum",
            image_interpolation="linear",
            include=include
        )
        # Translation
        tl = tio.RandomAffine(
            scales=(1.0, 1.0),
            degrees=0,
            translation=(15.0, 15.0, 0),
            isotropic=True,  # If True, scaling factor along all dimensions is the same
            center="image",
            default_pad_value="minimum",
            image_interpolation="linear",
            include=include,
        )

        # Bias Field
        bias_field = tio.transforms.RandomBiasField(
            coefficients=0.5, order=3, include=include_img
        )
        # Gaussian Noise
        random_noise_transform = tio.RandomNoise(
            mean=0,  # Default is 0, but you can adjust this if needed
            std=(0.01, 0.1),  # You can provide a range for standard deviation
            p=0.5,  # Probability of applying this transform
            include=include_img
        )

        all_augs = [
            rot,
            scaling,
            angf,
            tl,
            bias_field,
            random_noise_transform
        ]

        transform = tio.Compose(
            [tio.OneOf(all_augs, p=0.5)], include=include
        )

        dataset = dset.Dataset_t2(dataset_list=data_path,images_list=img_list,cfg=cfg,transforms=transform)
    else:
        data_path = cfg.DATA.PATH_HDF5_VAL
        shuffle = False
        logger.info(f"loading {mode.capitalize()} data ... from {data_path}")

        if cfg.MODEL.NUM_CHANNELS == 14:
            tensor_to_include = ToTensor_2input()
        elif cfg.MODEL.NUM_CHANNELS == 21:
            tensor_to_include = ToTensor_3input()
        else:
            tensor_to_include = ToTensor_1input()

        all_augs = [
            tensor_to_include,
        ]

        transform = tio.Compose(all_augs)

        dataset = dset.Dataset_V1(dataset_path=data_path, cfg=cfg, transforms=transform)

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
    cfg.merge_from_file('/home/uline/Desktop/white_matter_hyperintensities_segmen/config/FastSurferUNET_axial.yaml')

    train_loader = get_dataloader(cfg=cfg, mode='train')
    val_loader = get_dataloader(cfg=cfg, mode='val')

    next(iter(train_loader))

