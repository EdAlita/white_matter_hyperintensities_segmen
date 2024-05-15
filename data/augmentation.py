from numbers import Number, Real
from typing import Union, Tuple, Any, Dict
import numpy as np
import numpy.typing as npt
import torch
from torchvision import transforms
class ToTensor_1input(object):
    def __call__(self, sample: npt.NDArray) -> Dict[str, Any]:

        img, label, weight = (
            sample['img'],
            sample['label'],
            sample['weight']
        )

        img = img.astype(np.float32)
        img = np.clip(img / 255.0, a_min=0.0, a_max=1.0)
        img = img.transpose((2, 0, 1))

        return {
            "image": torch.from_numpy(img),
            "label": torch.from_numpy(label),
            "weight": torch.from_numpy(weight)
        }

class ToTensor_2input(object):
    def __call__(self, sample: npt.NDArray) -> Dict[str, Any]:

        img, img2, label, weight = (
            sample['img'],
            sample['img2'],
            sample['label'],
            sample['weight']
        )

        img = img.astype(np.float32)
        img = np.clip(img / 255.0, a_min=0.0, a_max=1.0)
        img = img.transpose((2, 0, 1))

        img2 = img2.astype(np.float32)
        img2 = np.clip(img2 / 255.0, a_min=0.0, a_max=1.0)
        img2 = img2.transpose((2, 0, 1))

        return {
            "image": torch.from_numpy(img),
            "image2": torch.from_numpy(img2),
            "label": torch.from_numpy(label),
            "weight": torch.from_numpy(weight)
        }


class ToTensor_3input(object):
    def __call__(self, sample: npt.NDArray) -> Dict[str, Any]:
        img, img2, img3, label, weight = (
            sample['img'],
            sample['img2'],
            sample['img3'],
            sample['label'],
            sample['weight']
        )

        img = img.astype(np.float32)
        img = np.clip(img / 255.0, a_min=0.0, a_max=1.0)
        img = img.transpose((2, 0, 1))

        img2 = img2.astype(np.float32)
        img2 = np.clip(img2 / 255.0, a_min=0.0, a_max=1.0)
        img2 = img2.transpose((2, 0, 1))

        img3 = img3.astype(np.float32)
        img3 = np.clip(img3 / 255.0, a_min=0.0, a_max=1.0)
        img3 = img3.transpose((2, 0, 1))

        return {
            "image": torch.from_numpy(img),
            "image2": torch.from_numpy(img2),
            "image3": torch.from_numpy(img3),
            "label": torch.from_numpy(label),
            "weight": torch.from_numpy(weight)
        }