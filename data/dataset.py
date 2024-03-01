import time
import h5py
from torch.utils.data import Dataset
from typing import Optional
import torch
import numpy as np
import yacs.config

from config.defaults import get_cfg_defaults
from utils.wm_logger import loggen

logger =loggen(__name__)


class Dataset_V1(Dataset):
    """
    Class for loading aseg file for trainnig
    """

    def __init__(
        self,
        dataset_path: str,
        cfg: yacs.config.CfgNode,
        transform: Optional = None):

        self.images = []
        self.labels = []
        self.subjects = []
        self.zooms = []

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

        img = img.astype(np.float32)
        img = np.clip(img / 255.0, a_min=0.0, a_max=1.0)
        img = img.transpose((2, 0, 1))

        return {
            "image": torch.from_numpy(img),
            "label": torch.from_numpy(label)
        }

    def __len__(self):
        """
        Get count.
        """
        return self.count

if __name__ == "__main__":

    train = Dataset_V1('/localmount/volume-hd/users/uline/fastCNN_split/axial/train_axial.hdf5')

    Dict = train.__getitem__(100)

    label = Dict["label"].numpy()
    img = Dict["image"].numpy()
    print(type(img),type(label))
    from matplotlib import pyplot as plt

    plt.imshow(img[4,:,:])
    plt.figure()
    plt.imshow(label)
    plt.show()