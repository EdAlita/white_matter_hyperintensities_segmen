import time
import h5py
from typing import List
from torch.utils.data import Dataset
from typing import Optional
import torch
import numpy as np
import yacs.config
import nibabel as nib
from tqdm import tqdm

from config.defaults import get_cfg_defaults
from utils.wm_logger import loggen
from data.data_utils import get_thick_slices, sagittal_transform_coronal, sagittal_transform_axial, \
    filter_blank_slices_thick, create_weight_mask

logger = loggen(__name__)


class Dataset_V1(Dataset):
    """
    Class for loading aseg file for trainnig
    """

    def __init__(
            self,
            dataset_path: str,
            cfg: yacs.config.CfgNode,
            transforms: Optional = None):

        self.images = []
        self.images2 = []
        self.images3 = []
        self.labels = []
        self.subjects = []
        self.zooms = []
        self.weights = []
        self.cfg = cfg
        self.transforms = transforms

        start = time.time()

        with h5py.File(dataset_path, "r") as hf:
            for size in hf.keys():
                try:
                    logger.info(f"Processing images of size {size}.")
                    img_dataset = hf[f"{size}"]["orig_dataset"]
                    logger.info(
                        "Processed volumes of size {} in {:.3f} seconds".format(
                            size, time.time() - start
                        )
                    )

                    self.images.extend(img_dataset)

                    if self.cfg.MODEL.NUM_CHANNELS == 14 or self.cfg.MODEL.NUM_CHANNELS == 21:
                        img_dataset2 = hf[f"{size}"]["orig2_dataset"]
                        logger.info(
                            "Processed 2nd volumes of size {} in {:.3f} seconds".format(
                                size, time.time() - start
                            )
                        )
                        self.images2.extend(img_dataset2)

                    if self.cfg.MODEL.NUM_CHANNELS == 21:
                        img_dataset3 = hf[f"{size}"]["orig3_dataset"]
                        logger.info(
                            "Processed 3rd volumes of size {} in {:.3f} seconds".format(
                                size, time.time() - start
                            )
                        )
                        self.images3.extend(img_dataset3)

                    self.labels.extend(list(hf[f"{size}"]["aseg_dataset"]))
                    logger.info(
                        "Processed segs of size {} in {:.3f} seconds".format(
                            size, time.time() - start
                        )
                    )
                    self.zooms.extend(list(hf[f"{size}"]["zoom_dataset"]))
                    logger.info(
                        "Processed zooms of size {} in {:.3f} seconds".format(
                            size, time.time() - start
                        )
                    )
                    self.subjects.extend(list(hf[f"{size}"]["subject"]))
                    logger.info(
                        "Processed subjects of size {} in {:.3f} seconds".format(
                            size, time.time() - start
                        )
                    )
                    self.weights.extend(list(hf[f"{size}"]["weight_dataset"]))
                    logger.info(
                        "Processed weights of size {} in {:.3f} seconds".format(
                            size, time.time() - start
                        )
                    )

                except KeyError as e:
                    print(
                        f"KeyError: Unable to open object (object {size} does not exist)"
                    )
                    continue

            self.count = len(self.images)

            logger.info(
                "Sucessfully loaded {} data from {} with plane {} in {:.3f} seconds".format(
                    self.count, dataset_path, self.cfg.DATA.PLANE, time.time() - start
                )
            )

    def get_subject_names(self):
        return self.subjects

    def __getitem__(self, index):

        img = self.images[index]
        label = self.labels[index]
        weight = self.weights[index]

        if self.cfg.MODEL.NUM_CHANNELS == 14:
            img2 = self.images2[index]

            if self.transforms is not None:
                tx_sample = self.transforms(
                    {
                        "img": img,
                        "img2": img2,
                        "label": label,
                        "weight": weight,
                    }
                )

            return {
                "image": tx_sample["image"],
                "image2": tx_sample["image2"],
                "label": tx_sample["label"],
                "weight": tx_sample["weight"]
            }
        elif self.cfg.MODEL.NUM_CHANNELS == 21:
            img2 = self.images2[index]
            img3 = self.images3[index]

            if self.transforms is not None:
                tx_sample = self.transforms(
                    {
                        "img": img,
                        "img2": img2,
                        "img3": img3,
                        "label": label,
                        "weight": weight,
                    }
                )

            return {
                "image": tx_sample["image"],
                "image2": tx_sample["image2"],
                "image3": tx_sample["image3"],
                "label": tx_sample["label"],
                "weight": tx_sample["weight"]
            }
        else:

            if self.transforms is not None:
                tx_sample = self.transforms(
                    {
                        "img": img,
                        "label": label,
                        "weight": weight,
                    }
                )

            return {
                "image": tx_sample["image"],
                "label": tx_sample["label"],
                "weight": tx_sample["weight"]
            }

    def __len__(self):
        """
        Get count.
        """
        return self.count


class Dataset_t1(Dataset):
    """
    Class for loading aseg file for trainnig
    """

    def __init__(
            self,
            dataset_path: str,
            cfg: yacs.config.CfgNode,
            transforms: Optional = None):

        self.images = []
        self.images2 = []
        self.images3 = []
        self.labels = []
        self.subjects = []
        self.zooms = []
        self.weights = []
        self.cfg = cfg
        self.transforms = transforms

        start = time.time()

        with h5py.File(dataset_path, "r") as hf:
            for size in hf.keys():
                try:
                    logger.info(f"Processing images of size {size}.")
                    img_dataset = hf[f"{size}"]["orig_dataset"]
                    logger.info(
                        "Processed volumes of size {} in {:.3f} seconds".format(
                            size, time.time() - start
                        )
                    )

                    self.images.extend(img_dataset)

                    if self.cfg.MODEL.NUM_CHANNELS == 14 or self.cfg.MODEL.NUM_CHANNELS == 21:
                        img_dataset2 = hf[f"{size}"]["orig2_dataset"]
                        logger.info(
                            "Processed 2nd volumes of size {} in {:.3f} seconds".format(
                                size, time.time() - start
                            )
                        )
                        self.images2.extend(img_dataset2)

                    if self.cfg.MODEL.NUM_CHANNELS == 21:
                        img_dataset3 = hf[f"{size}"]["orig3_dataset"]
                        logger.info(
                            "Processed 3rd volumes of size {} in {:.3f} seconds".format(
                                size, time.time() - start
                            )
                        )
                        self.images3.extend(img_dataset3)

                    self.labels.extend(list(hf[f"{size}"]["aseg_dataset"]))
                    logger.info(
                        "Processed segs of size {} in {:.3f} seconds".format(
                            size, time.time() - start
                        )
                    )
                    self.zooms.extend(list(hf[f"{size}"]["zoom_dataset"]))
                    logger.info(
                        "Processed zooms of size {} in {:.3f} seconds".format(
                            size, time.time() - start
                        )
                    )
                    self.subjects.extend(list(hf[f"{size}"]["subject"]))
                    logger.info(
                        "Processed subjects of size {} in {:.3f} seconds".format(
                            size, time.time() - start
                        )
                    )
                    self.weights.extend(list(hf[f"{size}"]["weight_dataset"]))
                    logger.info(
                        "Processed weights of size {} in {:.3f} seconds".format(
                            size, time.time() - start
                        )
                    )

                except KeyError as e:
                    print(
                        f"KeyError: Unable to open object (object {size} does not exist)"
                    )
                    continue

            self.count = len(self.images)

            logger.info(
                "Sucessfully loaded {} data from {} with plane {} in {:.3f} seconds".format(
                    self.count, dataset_path, self.cfg.DATA.PLANE, time.time() - start
                )
            )

    def get_subject_names(self):
        return self.subjects

    def __getitem__(self, index):

        img = self.images[index]
        label = self.labels[index]
        weight = self.weights[index]

        img = np.expand_dims(img.transpose((2, 0, 1)), axis=3)
        label = label[np.newaxis, :, :, np.newaxis]
        weight = weight[np.newaxis, :, :, np.newaxis]

        import torchio as tio

        subject = tio.Subject(
            {
                "img": tio.ScalarImage(tensor=img),
                "label": tio.LabelMap(tensor=label),
                "weight": tio.LabelMap(tensor=weight),
            }
        )

        if self.transforms is not None:
            tx_sample = self.transforms(subject)

        if self.cfg.MODEL.NUM_CHANNELS == 14:
            img2 = self.images2[index]
            img2 = np.expand_dims(img2.transpose((2, 0, 1)), axis=3)

            subject = tio.Subject(
                {
                    "img": tio.ScalarImage(tensor=img),
                    "img2": tio.ScalarImage(tensor=img2),
                    "label": tio.LabelMap(tensor=label),
                    "weight": tio.LabelMap(tensor=weight),
                }
            )

            if self.transforms is not None:
                tx_sample = self.transforms(subject)

            img = torch.squeeze(tx_sample["img"].data).float()
            img2 = torch.squeeze(tx_sample["img2"].data).float()
            label = torch.squeeze(tx_sample["label"].data).byte()
            weight = torch.squeeze(tx_sample["weight"].data).float()

            img = torch.clamp(img / 255.0, min=0.0, max=1.0)
            img2 = torch.clamp(img2 / 255.0, min=0.0, max=1.0)

            return {
                "image": img,
                "image2": img2,
                "label": label,
                "weight": weight
            }
        elif self.cfg.MODEL.NUM_CHANNELS == 21:
            img2 = self.images2[index]
            img3 = self.images3[index]
            img2 = np.expand_dims(img2.transpose((2, 0, 1)), axis=3)
            img3 = np.expand_dims(img3.transpose((2, 0, 1)), axis=3)

            subject = tio.Subject(
                {
                    "img": tio.ScalarImage(tensor=img),
                    "img2": tio.ScalarImage(tensor=img2),
                    "img3": tio.ScalarImage(tensor=img3),
                    "label": tio.LabelMap(tensor=label),
                    "weight": tio.LabelMap(tensor=weight),
                }
            )

            if self.transforms is not None:
                tx_sample = self.transforms(subject)

            img = torch.squeeze(tx_sample["img"].data).float()
            img2 = torch.squeeze(tx_sample["img2"].data).float()
            img3 = torch.squeeze(tx_sample["img3"].data).float()
            label = torch.squeeze(tx_sample["label"].data).byte()
            weight = torch.squeeze(tx_sample["weight"].data).float()

            img = torch.clamp(img / 255.0, min=0.0, max=1.0)
            img2 = torch.clamp(img2 / 255.0, min=0.0, max=1.0)
            img3 = torch.clamp(img3 / 255.0, min=0.0, max=1.0)
            return {
                "image": img,
                "image2": img2,
                "image3": img3,
                "label": label,
                "weight": weight
            }
        else:

            img = torch.squeeze(tx_sample["img"].data).float()
            label = torch.squeeze(tx_sample["label"].data).byte()
            weight = torch.squeeze(tx_sample["weight"].data).float()

            img = torch.clamp(img / 255.0, min=0.0, max=1.0)

            return {
                "image": img,
                "label": label,
                "weight": weight
            }

    def __len__(self):
        """
        Get count.
        """
        return self.count


class Dataset_t2(Dataset):
    """
    Class for loading aseg file for trainnig
    """

    def __init__(
            self,
            batch,
            cfg: yacs.config.CfgNode,
            transforms: Optional = None):
        self.images = []
        self.images2 = []
        self.images3 = []
        self.labels = []
        self.subjects = []
        self.zooms = []
        self.weights = []
        self.cfg = cfg
        self.transforms = transforms

        self.plane = cfg.DATA.PLANE
        if self.cfg.MODEL.NUM_CHANNELS == 14:
            volume, volume2, gt, weight = (
                    batch["image"],
                    batch["image2"],
                    batch["label"],
                    batch["weight"]
                )

            volume3 = volume

        else:
            volume, gt, weight = (
                batch["image"],
                batch["label"],
                batch["weight"]
            )
            volume3 = volume
            volume2 = volume

        volume, volume2, volume3, gt, weight = (
            volume.squeeze(1),
            volume2.squeeze(1),
            volume3.squeeze(1),
            gt.squeeze(1),
            weight.squeeze(1)
        )

        N, D, H, W = volume.shape

        volume, volume2, volume3, gt, weight = (
            volume.view(N * D, H, W),
            volume2.view(N * D, H, W),
            volume3.view(N * D, H, W),
            gt.view(N * D, H, W),
            weight.view(N * D, H, W)
        )

        volume, volume2, volume3, gt, weight = (
            volume.numpy(),
            volume2.numpy(),
            volume3.numpy(),
            gt.numpy(),
            weight.numpy()
        )

        if self.cfg.MODEL.NUM_CHANNELS == 14:
            thicknes = cfg.MODEL.NUM_CHANNELS // 4
        elif self.cfg.MODEL.NUM_CHANNELS == 21:
            thicknes = cfg.MODEL.NUM_CHANNELS // 6
        else:
            thicknes = cfg.MODEL.NUM_CHANNELS // 2

        volume_thick = get_thick_slices(volume, thicknes)
        volume2_thick = get_thick_slices(volume2, thicknes)
        volume3_thick = get_thick_slices(volume3, thicknes)

        filter_volume, filter_volume2, filter_volume3, filter_labels, filter_weight = filter_blank_slices_thick(
                volume_thick, volume2_thick, volume3_thick, gt, weight, threshold=10)

        self.images.extend(filter_volume)
        self.images2.extend(filter_volume2)
        self.images3.extend(filter_volume3)
        self.labels.extend(filter_labels)
        self.weights.extend(filter_weight)

        self.count = len(self.images)

    def __getitem__(self, index):

        img = self.images[index]
        label = self.labels[index]
        weight = self.weights[index]

        img = np.expand_dims(img.transpose((2, 0, 1)), axis=3)
        label = label[np.newaxis, :, :, np.newaxis]
        weight = weight[np.newaxis, :, :, np.newaxis]

        import torchio as tio

        subject = tio.Subject(
            {
                "img": tio.ScalarImage(tensor=img),
                "label": tio.LabelMap(tensor=label),
                "weight": tio.LabelMap(tensor=weight),
            }
        )

        if self.transforms is not None:
            tx_sample = self.transforms(subject)

        if self.cfg.MODEL.NUM_CHANNELS == 14:
            img2 = self.images2[index]
            img2 = np.expand_dims(img2.transpose((2, 0, 1)), axis=3)

            subject = tio.Subject(
                {
                    "img": tio.ScalarImage(tensor=img),
                    "img2": tio.ScalarImage(tensor=img2),
                    "label": tio.LabelMap(tensor=label),
                    "weight": tio.LabelMap(tensor=weight),
                }
            )

            if self.transforms is not None:
                tx_sample = self.transforms(subject)

            img = torch.squeeze(tx_sample["img"].data).float()
            img2 = torch.squeeze(tx_sample["img2"].data).float()
            label = torch.squeeze(tx_sample["label"].data).byte()
            weight = torch.squeeze(tx_sample["weight"].data).float()

            img = torch.clamp(img / 255.0, min=0.0, max=1.0)
            img2 = torch.clamp(img2 / 255.0, min=0.0, max=1.0)

            return {
                "image": img,
                "image2": img2,
                "label": label,
                "weight": weight
            }
        elif self.cfg.MODEL.NUM_CHANNELS == 21:
            img2 = self.images2[index]
            img3 = self.images3[index]
            img2 = np.expand_dims(img2.transpose((2, 0, 1)), axis=3)
            img3 = np.expand_dims(img3.transpose((2, 0, 1)), axis=3)

            subject = tio.Subject(
                {
                    "img": tio.ScalarImage(tensor=img),
                    "img2": tio.ScalarImage(tensor=img2),
                    "img3": tio.ScalarImage(tensor=img3),
                    "label": tio.LabelMap(tensor=label),
                    "weight": tio.LabelMap(tensor=weight),
                }
            )

            if self.transforms is not None:
                tx_sample = self.transforms(subject)

            img = torch.squeeze(tx_sample["img"].data).float()
            img2 = torch.squeeze(tx_sample["img2"].data).float()
            img3 = torch.squeeze(tx_sample["img3"].data).float()
            label = torch.squeeze(tx_sample["label"].data).byte()
            weight = torch.squeeze(tx_sample["weight"].data).float()

            img = torch.clamp(img / 255.0, min=0.0, max=1.0)
            img2 = torch.clamp(img2 / 255.0, min=0.0, max=1.0)
            img3 = torch.clamp(img3 / 255.0, min=0.0, max=1.0)
            return {
                "image": img,
                "image2": img2,
                "image3": img3,
                "label": label,
                "weight": weight
            }
        else:

            img = torch.squeeze(tx_sample["img"].data).float()
            label = torch.squeeze(tx_sample["label"].data).byte()
            weight = torch.squeeze(tx_sample["weight"].data).float()

            img = torch.clamp(img / 255.0, min=0.0, max=1.0)

            return {
                "image": img,
                "label": label,
                "weight": weight
            }

    def __len__(self):
        """
        Get count.
        """
        return self.count


class cases_dataset(Dataset):
    def __init__(self,
                 dataset: List,
                 images_list: List,
                 cfg: yacs.config.CfgNode
                 ):
        self.dataset = dataset
        self.images_list = images_list
        self.cfg = cfg
        self.plane = cfg.DATA.PLANE

    def __getitem__(self, index):
        volume_img = nib.load(str(self.dataset[index] / self.images_list[0]))
        gt_img = nib.load(str(self.dataset[index] / 'lesion.nii.gz'))

        if self.cfg.MODEL.NUM_CHANNELS == 14:
            volume2_img = nib.load(str(self.dataset[index] / self.images_list[1]))
            volume3_img = nib.load(str(self.dataset[index] / self.images_list[1]))
        elif self.cfg.MODEL.NUM_CHANNELS == 21:
            volume2_img = nib.load(str(self.dataset[index] / self.images_list[1]))
            volume3_img = nib.load(str(self.dataset[index] / self.images_list[2]))
        else:
            volume2_img = nib.load(str(self.dataset[index] / self.images_list[0]))
            volume3_img = nib.load(str(self.dataset[index] / self.images_list[0]))

        volume_img = nib.as_closest_canonical(volume_img)
        volume2_img = nib.as_closest_canonical(volume2_img)
        volume3_img = nib.as_closest_canonical(volume3_img)
        gt_img = nib.as_closest_canonical(gt_img)

        gt = np.asarray(
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

        if self.plane == "axial":
            volume = sagittal_transform_axial(volume)
            volume2 = sagittal_transform_axial(volume2)
            volume3 = sagittal_transform_axial(volume3)
            gt = sagittal_transform_axial(gt)

        if self.plane == "coronal":
            volume = sagittal_transform_coronal(volume)
            volume2 = sagittal_transform_coronal(volume2)
            volume3 = sagittal_transform_coronal(volume3)
            gt = sagittal_transform_coronal(gt)

        weight = create_weight_mask(
            gt,
            max_weight=5,
            max_edge_weight=5,
            gradient=True
        )

        if self.cfg.MODEL.NUM_CHANNELS == 14:
            return {
                "image": volume,
                "image2": volume2,
                "label": gt,
                "weight": weight
            }
        else:
            return {
                "image": volume,
                "label": gt,
                "weight": weight
            }

    def __len__(self):
        return len(self.dataset)


class data_set_from_origdata(Dataset):
    def __init__(self,
                 dataset: List,
                 cfg: yacs.config.CfgNode,
                 ):
        self.dataset = dataset
        self.cfg = cfg

        self.plane = cfg.DATA.PLANE

        volume_img = nib.load(str(self.dataset[0]))

        if self.cfg.MODEL.NUM_CHANNELS == 14:
            volume2_img = nib.load(str(self.dataset[1]))
            volume3_img = nib.load(str(self.dataset[1]))
            thicknes = cfg.MODEL.NUM_CHANNELS // 4
        elif self.cfg.MODEL.NUM_CHANNELS == 21:
            volume2_img = nib.load(str(self.dataset[1]))
            volume3_img = nib.load(str(self.dataset[2]))
            thicknes = cfg.MODEL.NUM_CHANNELS // 6
        else:
            volume2_img = nib.load(str(self.dataset[0]))
            volume3_img = nib.load(str(self.dataset[0]))
            thicknes = cfg.MODEL.NUM_CHANNELS // 2

        self.header = volume_img.header
        self.affine = volume_img.affine

        volume_img = nib.as_closest_canonical(volume_img)
        volume2_img = nib.as_closest_canonical(volume2_img)
        volume3_img = nib.as_closest_canonical(volume3_img)

        volume = np.asarray(
            volume_img.get_fdata(), dtype=np.uint8
        )

        volume2 = np.asarray(
            volume2_img.get_fdata(), dtype=np.uint8
        )

        volume3 = np.asarray(
            volume3_img.get_fdata(), dtype=np.uint8
        )

        if self.plane == "axial":
            volume = sagittal_transform_axial(volume)
            volume2 = sagittal_transform_axial(volume2)
            volume3 = sagittal_transform_axial(volume3)

        if self.plane == "coronal":
            volume = sagittal_transform_coronal(volume)
            volume2 = sagittal_transform_coronal(volume2)
            volume3 = sagittal_transform_coronal(volume3)

        volume_thick = get_thick_slices(volume, thicknes)
        volume2_thick = get_thick_slices(volume2, thicknes)
        volume3_thick = get_thick_slices(volume3, thicknes)

        self.images = volume_thick
        self.images2 = volume2_thick
        self.images3 = volume3_thick
        self.count = len(self.images)

    def get_img_info(self):
        return self.affine, self.header

    def __getitem__(self, index):

        img = self.images[index]

        img = img.astype(np.float32)
        img = np.clip(img / 255.0, a_min=0.0, a_max=1.0)
        img = img.transpose((2, 0, 1))

        if self.cfg.MODEL.NUM_CHANNELS == 14:
            img2 = self.images2[index]

            img2 = img2.astype(np.float32)
            img2 = np.clip(img2 / 255.0, a_min=0.0, a_max=1.0)
            img2 = img2.transpose((2, 0, 1))
            return {
                "image": torch.from_numpy(img),
                "image2": torch.from_numpy(img2)
            }
        elif self.cfg.MODEL.NUM_CHANNELS == 21:
            img2 = self.images2[index]

            img2 = img2.astype(np.float32)
            img2 = np.clip(img2 / 255.0, a_min=0.0, a_max=1.0)
            img2 = img2.transpose((2, 0, 1))

            img3 = self.images3[index]

            img3 = img3.astype(np.float32)
            img3 = np.clip(img3 / 255.0, a_min=0.0, a_max=1.0)
            img3 = img3.transpose((2, 0, 1))
            return {
                "image": torch.from_numpy(img),
                "image2": torch.from_numpy(img2),
                "image3": torch.from_numpy(img3)
            }
        else:
            return {
                "image": torch.from_numpy(img)
            }

    def __len__(self):
        """
        Get count.
        """
        return self.count


if __name__ == "__main__":
    cfg = get_cfg_defaults()
    train = Dataset_V1('/localmount/volume-ssd/users/uline/input_test/data/train_axial_FlairT2.hdf5', cfg)

    Dict = train.__getitem__(200)

    label = Dict["label"].numpy()
    img = Dict["image"].numpy()
    img2 = Dict["image2"].numpy()
    print(img.shape, img2.shape)
    from matplotlib import pyplot as plt

    plt.imshow(img[4, :, :], cmap='gray')
    plt.figure()
    plt.imshow(img2[4, :, :], cmap='gray')
    plt.show()
