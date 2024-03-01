import time

import h5py

from torch.utils.data import Dataset
from typing import Optional, Turple, Dict

import torch
from utils import logging

import yacs.config

logger = logging.getLogger(__name__)


class Dataset_train(Dataset):
    def __init__(self,
                 dataset_path: str,
                 cfg: yacs.config.CGFNode,
                 transform: Optional = None):
        self.max_size = cfg.DATA.PADDED_SIZE
        self.base_res = cfg.MODEL.BASE_RES

        self.images = []
        self.labels = []
        self.subjects = []
        self.zooms = []

        start = time.time()

        with h5py.File(dataset_path, "r") as hf:
            for size in cfg.DATA.SIZES:
                try:
                    logger.info(f"Processing images of size {size}.")
                    img_dataset = list(hf[f"{size}"]["volume"])
                    logger.info(
                        "Processed volumes of size {} in {:.3f} seconds".format(
                            size, time.time() - start
                        )
                    )

                    self.images.extend(img_dataset)
                    self.labels.extend(list(hf[f"{size}"]["seg"]))
                    logger.info(
                        "Processed segs of size {} in {:.3f} seconds".format(
                            size, time.time() - start
                        )
                    )
                    self.zooms.extend(list(hf[f"{size}"]["zoom"]))
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

                except KeyError as e:
                    print(
                        f"KeyError: Unable to open object (object {size} does not exist)"
                    )
                    continue

            self.count = len(self.images)
            self.transform = transform

            logger.info(
                "Sucessfully loaded {} data from {} with plane {} in {:.3f} seconds".format(
                    self.count, dataset_path, cfg.DATA.PLANE, time.time() - start
                )
            )

    def get_subject_names(self):
        return self.subjects

    def __getitem__(self, index):

        import torchio as tio
        subject = tio.Subject(
            {
                "img": tio.ScalarImage(tensor=self.images[index]),
                "label": tio.LabelMap(tensor=self.labels[index])
            }
        )

        if self.transform is not None:
            tx_sample = self.transform(subject)  # this returns data as torch.tensors

            img = torch.squeeze(tx_sample["img"].data).float()
            label = torch.squeeze(tx_sample["label"].data).byte()

            # Normalize image and clamp between 0 and 1
            img = torch.clamp(img / img.max(), min=0.0, max=1.0)

        return {
            "image": img,
            "label": label
        }

    def __len__(self):
        """
        Return count.
        """
        return self.count




class DatasetVal(Dataset):
    """
    Class for loading aseg file for trainnig
    """

    def __init__(
        self,
        dataset_path: str,
        cfg: yacs.config.CfgNode,
        transform: Optional = None):
        self.max_size = cfg.DATA.PADDED_SIZE
        self.base_res = cfg.MODEL.BASE_RES

        self.images = []
        self.labels = []
        self.subjects = []
        self.zooms = []

        start = time.time()

    
        with h5py.File(dataset_path, "r") as hf:
            for size in cfg.DATA.SIZES:
                try:
                    logger.info(f"Processing images of size {size}.")
                    img_dataset = list(hf[f"{size}"]["volume"])
                    logger.info(
                        "Processed volumes of size {} in {:.3f} seconds".format(
                            size, time.time() - start
                        )
                    )

                    self.images.extend(img_dataset)
                    self.labels.extend(list(hf[f"{size}"]["seg"]))
                    logger.info(
                            "Processed segs of size {} in {:.3f} seconds".format(
                            size, time.time() - start
                        )
                    )
                    self.zooms.extend(list(hf[f"{size}"]["zoom"]))
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

                except KeyError as e:
                    print(
                         f"KeyError: Unable to open object (object {size} does not exist)"
                    )
                    continue

            self.count = len(self.images)
            self.transform = transform

            logger.info(
                "Sucessfully loaded {} data from {} with plane {} in {:.3f} seconds".format(
                    self.count, dataset_path, cfg.DATA.PLANE, time.time() - start
                )
            )

    def get_subject_names(self):
        return self.subjects

    def __getitem__(self, index):

        img = self.images[index]
        label = self.labels[index]

        if self.transform is not None:
            tx_sample = self.transform(
                {
                    "img": img,
                    "label": label
                }
            )

            img = tx_sample["img"]
            label = tx_sample["label"]

        return {
            "image": img,
            "label": label
        }

    def __len__(self):
        return self.count
                



        
