#!/bin/bash

echo -e "\e[1;37;41m Trainning nnUnet \e[0m"

echo -e "\e[1;37;41m Activating virtual env \e[0m"

source /home/uline/Desktop/nnUnet/.venv_nnunet/bin/activate

echo -e "\e[1;37;41m inference flair \e[0m"

nnUNet_predict -i /localmount/volume-ssd/users/uline/nnUNet_CVD_input_test/data/nnUNet_raw/nnUNet_raw_data/Task019_flairbiobankaxial/miccai_2017_ts -o /localmount/volume-hd/users/uline/segmentation_results/miccai_2017/nnUNet/flair/out/axial -m 2d -tr nnUNetTrainerV2_Fast -t 001 -f 0 -z

nnUNet_predict -i /localmount/volume-ssd/users/uline/nnUNet_CVD_input_test/data/nnUNet_raw/nnUNet_raw_data/Task020_flairbiobankcoronal/miccai_2017_ts -o /localmount/volume-hd/users/uline/segmentation_results/miccai_2017/nnUNet/flair/out/coronal -m 2d -tr nnUNetTrainerV2_Fast -t 002 -f 0 -z

nnUNet_predict -i /localmount/volume-ssd/users/uline/nnUNet_CVD_input_test/data/nnUNet_raw/nnUNet_raw_data/Task021_flairbiobanksagital/miccai_2017_ts -o /localmount/volume-hd/users/uline/segmentation_results/miccai_2017/nnUNet/flair/out/sagital -m 2d -tr nnUNetTrainerV2_Fast -t 003 -f 0 -z

deactivate

echo -e "\e[1;37;41m Finish Script \e[0m"
