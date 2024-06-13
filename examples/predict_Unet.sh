#!/bin/bash

dir=flair
file=FLAIR
num=7

kernel=3x3

cd miccai_2017

cd flair_best_config

echo -e "\e[1;37;41m Running UNET axial inference $kernel \e[0m"

mkdir out/UNET_axial

python3 /home/uline/Desktop/white_matter_hyperintensities_segmen/inference.py --dataset_path /localmount/volume-hd/users/uline/data_sets//MICCAI_2017 \
--volume_name $file.nii.gz \
--cfg_file /home/uline/Desktop/white_matter_hyperintensities_segmen/config/FastSurferUNET_axial.yaml \
--csv_file /localmount/volume-hd/users/uline/data_sets//MICCAI_2017/wmh_overall.csv --plane axial --out_path out/UNET_axial \
--ckpt_path /localmount/volume-hd/users/uline/segmentation_results/aug_test/$dir/FastSurferUNET/checkpoints/FastSurferUNET_axial/Best_training_state.pkl \
--num_channels $num

echo -e "\e[1;37;41m Running UNET coronal inference $kernel \e[0m"

mkdir out/UNET_coronal

python3 /home/uline/Desktop/white_matter_hyperintensities_segmen/inference.py --dataset_path /localmount/volume-hd/users/uline/data_sets//MICCAI_2017 \
--volume_name $file.nii.gz \
--cfg_file /home/uline/Desktop/white_matter_hyperintensities_segmen/config/FastSurferUNET_coronal.yaml \
--csv_file /localmount/volume-hd/users/uline/data_sets//MICCAI_2017/wmh_overall.csv --plane coronal --out_path out/UNET_coronal \
--ckpt_path /localmount/volume-hd/users/uline/segmentation_results/aug_test/$dir/FastSurferUNET/checkpoints/FastSurferUNET_coronal/Best_training_state.pkl \
--num_channels $num


echo -e "\e[1;37;41m Running UNET sagital inference $kernel \e[0m"

mkdir out/UNET_sagital

python3 /home/uline/Desktop/white_matter_hyperintensities_segmen/inference.py --dataset_path /localmount/volume-hd/users/uline/data_sets//MICCAI_2017 \
--volume_name $file.nii.gz \
--cfg_file /home/uline/Desktop/white_matter_hyperintensities_segmen/config/FastSurferUNET_sagital.yaml \
--csv_file /localmount/volume-hd/users/uline/data_sets//MICCAI_2017/wmh_overall.csv --plane sagital --out_path out/UNET_sagital \
--ckpt_path /localmount/volume-hd/users/uline/segmentation_results/aug_test/$dir/FastSurferUNET/checkpoints/FastSurferUNET_sagital/Best_training_state.pkl \
--num_channels $num


echo -e "\e[1;37;41m Running CNN axial inference $kernel \e[0m"

mkdir out/CNN_axial

python3 /home/uline/Desktop/white_matter_hyperintensities_segmen/inference.py --dataset_path /localmount/volume-hd/users/uline/data_sets//MICCAI_2017 \
--volume_name $file.nii.gz \
--cfg_file /home/uline/Desktop/white_matter_hyperintensities_segmen/config/FastSurferCNN_axial.yaml \
--csv_file /localmount/volume-hd/users/uline/data_sets//MICCAI_2017/wmh_overall.csv --plane axial --out_path out/CNN_axial \
--ckpt_path /localmount/volume-hd/users/uline/segmentation_results/aug_test/$dir/FastSurferCNN/checkpoints/FastSurferCNN_axial/Best_training_state.pkl \
--num_channels $num
 
echo -e "\e[1;37;41m Running CNN coronal inference $kernel \e[0m"

mkdir out/CNN_coronal

python3 /home/uline/Desktop/white_matter_hyperintensities_segmen/inference.py --dataset_path /localmount/volume-hd/users/uline/data_sets//MICCAI_2017 \
--volume_name $file.nii.gz \
--cfg_file /home/uline/Desktop/white_matter_hyperintensities_segmen/config/FastSurferCNN_coronal.yaml \
--csv_file /localmount/volume-hd/users/uline/data_sets//MICCAI_2017/wmh_overall.csv --plane coronal --out_path out/CNN_coronal \
--ckpt_path /localmount/volume-hd/users/uline/segmentation_results/aug_test/$dir/FastSurferCNN/checkpoints/FastSurferCNN_coronal/Best_training_state.pkl \
--num_channels $num


echo -e "\e[1;37;41m Running CNN sagital inference $kernel \e[0m"

mkdir out/CNN_sagital

python3 /home/uline/Desktop/white_matter_hyperintensities_segmen/inference.py --dataset_path /localmount/volume-hd/users/uline/data_sets//MICCAI_2017 \
--volume_name $file.nii.gz \
--cfg_file /home/uline/Desktop/white_matter_hyperintensities_segmen/config/FastSurferCNN_sagital.yaml \
--csv_file /localmount/volume-hd/users/uline/data_sets//MICCAI_2017/wmh_overall.csv --plane sagital --out_path out/CNN_sagital \
--ckpt_path /localmount/volume-hd/users/uline/segmentation_results/aug_test/$dir/FastSurferCNN/checkpoints/FastSurferCNN_sagital/Best_training_state.pkl \
--num_channels $num

echo -e "\e[1;37;41m Finish Script \e[0m" 







