from pathlib import Path

import torch
from torch.utils.data import DataLoader
from config.defaults import get_cfg_defaults
import time
import numpy as np
import yacs.config
from typing import Dict
from tqdm import tqdm
import pandas as pd
import nibabel as nib



from data.data_utils import sagittal_transform_axial, sagittal_transform_coronal
from models.networks import build_model
from data.dataset import data_set_from_origdata
from utils import wm_logger

logger = wm_logger.loggen(__name__)

class Inference:
    def __init__(self,
                 params: Dict,
                 cfg: yacs.config.CfgNode
                 ):
        self.dataset_path = Path(params["dataset_path"])
        self.volume_name = params["volume_name"]
        self.volume2_name = params["volume2_name"]
        self.volume3_name = params["volume3_name"]
        self.csv_file = Path(params["csv_file"])
        self.plane = params["plane"]
        self.datatype = "test"
        self.out_dir = Path(params["out_path"])
        self.ckpt = params["ckpt_path"]

        assert self.dataset_path.is_dir(), f"The provided paths are not valid: {self.dataset_path}!"

        dataframe = pd.read_csv(self.csv_file)
        dataframe = dataframe[dataframe["wmh_split"].str.contains(self.datatype)]

        self.subjects_dirs = [ self.dataset_path / row["imageid"] for index, row in dataframe.iterrows() ]

        logger.info(f"Loading {len(self.subjects_dirs)} for inference")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Using device: {self.device} for inference")

        np.random.seed(cfg.RNG_SEED)
        torch.manual_seed(cfg.RNG_SEED)
        self.cfg = cfg

        self.data_set_size = len(dataframe)

        self.model = build_model(cfg)

        model_state = torch.load(self.ckpt, map_location=self.device)
        self.model.load_state_dict(model_state["model_state"])

        self.model = self.model.to(self.device)

        logger.info(f"Loading {self.ckpt} with model {cfg.MODEL.MODEL_NAME}")

    @torch.no_grad()
    def get_prediction(self):
        self.model.eval()

        for idx, current_subject in enumerate(self.subjects_dirs):

                start = time.time()

                logger.info(
                    f"Volume Nr: {idx + 1} getting inference from {current_subject.name}/{self.volume_name}"
                )

                if self.cfg.MODEL.NUM_CHANNELS == 14:
                    list_input = [current_subject / self.volume_name,
                                  current_subject / self.volume2_name]
                    dataset = data_set_from_origdata(list_input, self.cfg)
                elif self.cfg.MODEL.NUM_CHANNELS == 21:
                    list_input = [current_subject / self.volume_name,
                                  current_subject / self.volume2_name,
                                  current_subject / self.volume3_name]
                    dataset = data_set_from_origdata(list_input, self.cfg)
                else:
                    list_input = [current_subject / self.volume_name]
                    dataset = data_set_from_origdata(list_input,self.cfg)

                dataloader = DataLoader(
                        dataset=dataset,
                        batch_size=self.cfg.TEST.BATCH_SIZE,
                        num_workers=self.cfg.TRAIN.NUM_WORKERS,
                        shuffle=False,
                        pin_memory=True)

                index = 0
                out = np.zeros([256,256,256])
                out2 = np.zeros([256,2,256,256])
                for curr_iter, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
                    images = batch["image"].to(self.device)

                    if self.cfg.MODEL.NUM_CHANNELS == 14:
                        images2 = batch["image2"].to(self.device)
                        inputs = torch.cat((images, images2), dim=1)
                    elif self.cfg.MODEL.NUM_CHANNELS == 21:
                        images2 = batch["image2"].to(self.device)
                        images3 = batch["image3"].to(self.device)
                        inputs = torch.cat((images, images2, images3), dim=1)
                    else:
                        inputs = images
                    pred = self.model(inputs)

                    pred_classes = torch.argmax(pred,dim=1)
                    prob = torch.nn.functional.softmax(pred, dim=1)

                    new_index = index+self.cfg.TEST.BATCH_SIZE

                    out[index:new_index, :, :] = pred_classes.cpu().numpy()
                    out2[index:new_index, :, :,:] = prob.cpu().numpy()

                    index = new_index

                if self.plane == "axial":
                    volume = sagittal_transform_axial(out,inverse=True)
                elif self.plane == "coronal":
                    volume = sagittal_transform_coronal(out,inverse=True)
                else:
                    volume = out

                affine, header = dataset.get_img_info()

                new_img = nib.Nifti1Image(np.asarray((volume > 0.0) * 1.0,np.uint8), affine, header)

                logger.info(f"Saving {self.out_dir} / f'{current_subject.name}.nii.gz'")

                nib.save(new_img, self.out_dir / f'{current_subject.name}.nii.gz')

                logger.info(f"Saving {self.out_dir} / f'{current_subject.name}.npz'")
                np.savez(self.out_dir / f'{current_subject.name}.npz',softmax=out2)

                logger.info(
                    "Inference for {} took: {:.3f} seconds".format(
                        current_subject.name, time.time() - start
                    ))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Inference algorithm")

    parser.add_argument("--dataset_path",
    type=str,
    help="name of the dataset to load for inference")

    parser.add_argument("--volume_name",
    type=str,
    help="name of the volume to load for inference")

    parser.add_argument("--volume2_name",
    type=str,
    help="name of the volume to load for inference",
    default=None)

    parser.add_argument("--volume3_name",
    type=str,
    help="name of the volume to load for inference",
    default=None)

    parser.add_argument("--cfg_file",
    type=str,
    help="path for the cfg file")

    parser.add_argument("--csv_file",
    type=str,
    help="path for the csv file describing the dataset")

    parser.add_argument("--plane",
    type=str,
    help="name of the plane to load in inference")

    parser.add_argument("--out_path",
    type=str,
    help="outpath to save the volumes")

    parser.add_argument("--ckpt_path", type=str, help="path to ckpt to load")

    parser.add_argument("--num_channels", type=int, help="channels for the input")

    args = parser.parse_args()

    dataset_params = {
        "dataset_path": args.dataset_path,
        "volume_name":args.volume_name,
        "volume2_name": args.volume2_name,
        "volume3_name": args.volume3_name,
        "csv_file":args.csv_file,
        "plane":args.plane,
        "out_path":args.out_path,
        "ckpt_path": args.ckpt_path
    }

    cfg = get_cfg_defaults()

    cfg.merge_from_file(args.cfg_file)

    cfg.MODEL.NUM_CHANNELS = args.num_channels

    inf = Inference(params=dataset_params, cfg=cfg)

    inf.get_prediction()
