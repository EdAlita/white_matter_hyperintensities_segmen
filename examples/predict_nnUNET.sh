#! /bin/bash

echo -e "\e[1;37;41m Activating virtual env \e[0m"

source /home/uline/Desktop/nnUnet/.venv_nnunet/bin/activate

echo -e "\e[1;37;41m running t1flair \e[0m"

nnUNet_predict -i /localmount/volume-ssd/users/uline/nnUNet_CVD_input_test/data/nnUNet_raw/nnUNet_raw_data/Task010_t1flairaxial/imagesTs -o /localmount/volume-hd/users/uline/segmentation_results/inputtestnnUnet/t1flair/out_test/axial -m 2d -tr nnUNetTrainerV2_Fast -t 010 -f 0 -z

nnUNet_predict -i /localmount/volume-ssd/users/uline/nnUNet_CVD_input_test/data/nnUNet_raw/nnUNet_raw_data/Task011_t1flaircoronal/imagesTs -o /localmount/volume-hd/users/uline/segmentation_results/inputtestnnUnet/t1flair/out_test/coronal -m 2d -tr nnUNetTrainerV2_Fast -t 011 -f 0 -z

nnUNet_predict -i /localmount/volume-ssd/users/uline/nnUNet_CVD_input_test/data/nnUNet_raw/nnUNet_raw_data/Task012_t1flairsagital/imagesTs -o /localmount/volume-hd/users/uline/segmentation_results/inputtestnnUnet/t1flair/out_test/sagital -m 2d -tr nnUNetTrainerV2_Fast -t 012 -f 0 -z


deactivate

echo -e "\e[1;37;41m finishing script \e[0m"
