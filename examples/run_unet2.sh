#!/bin/bash

dir=Flair
num_c=7

echo -e "\e[1;37;41m Running UNET axial training $dir \e[0m"

cd uk-biobank

cd $dir

python3 /home/uline/Desktop/white_matter_hyperintensities_segmen/train.py --config_path /home/uline/Desktop/white_matter_hyperintensities_segmen/config/FastSurferUNET_axial.yaml --train_path /localmount/volume-ssd/users/uline/transfer/test_axial_Flair.hdf5 --val_path /localmount/volume-ssd/users/uline/transfer/val_axial_Flair.hdf5 --num_channels $num_c
 

echo -e "\e[1;37;41m Running UNET coronal training $dir \e[0m"

python3 /home/uline/Desktop/white_matter_hyperintensities_segmen/train.py --config_path /home/uline/Desktop/white_matter_hyperintensities_segmen/config/FastSurferUNET_coronal.yaml --train_path /localmount/volume-ssd/users/uline/transfer/test_coronal_Flair.hdf5 --val_path /localmount/volume-ssd/users/uline/transfer/val_coronal_Flair.hdf5 --num_channels $num_c


echo -e "\e[1;37;41m Running UNET sagital training $dir \e[0m"

python3 /home/uline/Desktop/white_matter_hyperintensities_segmen/train.py --config_path /home/uline/Desktop/white_matter_hyperintensities_segmen/config/FastSurferUNET_sagital.yaml --train_path /localmount/volume-ssd/users/uline/transfer/test_sagital_Flair.hdf5 --val_path //localmount/volume-ssd/users/uline/transfer/val_sagital_Flair.hdf5 --num_channels $num_c

echo -e "\e[1;37;41m Running CNN axial training $dir \e[0m"

python3 /home/uline/Desktop/white_matter_hyperintensities_segmen/train.py --config_path /home/uline/Desktop/white_matter_hyperintensities_segmen/config/FastSurferCNN_axial.yaml --train_path /localmount/volume-ssd/users/uline/transfer/test_axial_Flair.hdf5 --val_path /localmount/volume-ssd/users/uline/transfer/val_axial_Flair.hdf5 --num_channels $num_c
 

echo -e "\e[1;37;41m Running CNN coronal training $dir \e[0m"

python3 /home/uline/Desktop/white_matter_hyperintensities_segmen/train.py --config_path /home/uline/Desktop/white_matter_hyperintensities_segmen/config/FastSurferCNN_coronal.yaml  --train_path /localmount/volume-ssd/users/uline/transfer/test_coronal_Flair.hdf5 --val_path /localmount/volume-ssd/users/uline/transfer/val_coronal_Flair.hdf5 --num_channels $num_c

echo -e "\e[1;37;41m Running CNN sagital training $dir \e[0m"

python3 /home/uline/Desktop/white_matter_hyperintensities_segmen/train.py --config_path /home/uline/Desktop/white_matter_hyperintensities_segmen/config/FastSurferCNN_sagital.yaml --train_path /localmount/volume-ssd/users/uline/transfer/test_sagital_Flair.hdf5 --val_path /localmount/volume-ssd/users/uline/transfer/val_sagital_Flair.hdf5 --num_channels $num_c

cd ..

dir=FlairT1
num_c=14

echo -e "\e[1;37;41m Running UNET axial training $dir \e[0m"

cd $dir

python3 /home/uline/Desktop/white_matter_hyperintensities_segmen/train.py --config_path /home/uline/Desktop/white_matter_hyperintensities_segmen/config/FastSurferUNET_axial.yaml --train_path /localmount/volume-ssd/users/uline/transfer/test_axial_FlairT1.hdf5 --val_path /localmount/volume-ssd/users/uline/transfer/val_axial_FlairT1.hdf5 --num_channels $num_c
 

echo -e "\e[1;37;41m Running UNET coronal training $dir \e[0m"

python3 /home/uline/Desktop/white_matter_hyperintensities_segmen/train.py --config_path /home/uline/Desktop/white_matter_hyperintensities_segmen/config/FastSurferUNET_coronal.yaml --train_path /localmount/volume-ssd/users/uline/transfer/test_coronal_FlairT1.hdf5 --val_path /localmount/volume-ssd/users/uline/transfer/val_coronal_FlairT1.hdf5 --num_channels $num_c


echo -e "\e[1;37;41m Running UNET sagital training $dir \e[0m"

python3 /home/uline/Desktop/white_matter_hyperintensities_segmen/train.py --config_path /home/uline/Desktop/white_matter_hyperintensities_segmen/config/FastSurferUNET_sagital.yaml --train_path /localmount/volume-ssd/users/uline/transfer/test_sagital_FlairT1.hdf5 --val_path //localmount/volume-ssd/users/uline/transfer/val_sagital_FlairT1.hdf5 --num_channels $num_c

echo -e "\e[1;37;41m Running CNN axial training $dir \e[0m"

python3 /home/uline/Desktop/white_matter_hyperintensities_segmen/train.py --config_path /home/uline/Desktop/white_matter_hyperintensities_segmen/config/FastSurferCNN_axial.yaml --train_path /localmount/volume-ssd/users/uline/transfer/test_axial_FlairT1.hdf5 --val_path /localmount/volume-ssd/users/uline/transfer/val_axial_FlairT1.hdf5 --num_channels $num_c
 

echo -e "\e[1;37;41m Running CNN coronal training $dir \e[0m"

python3 /home/uline/Desktop/white_matter_hyperintensities_segmen/train.py --config_path /home/uline/Desktop/white_matter_hyperintensities_segmen/config/FastSurferCNN_coronal.yaml  --train_path /localmount/volume-ssd/users/uline/transfer/test_coronal_FlairT1.hdf5 --val_path /localmount/volume-ssd/users/uline/transfer/val_coronal_FlairT1.hdf5 --num_channels $num_c

echo -e "\e[1;37;41m Running CNN sagital training $dir \e[0m"

python3 /home/uline/Desktop/white_matter_hyperintensities_segmen/train.py --config_path /home/uline/Desktop/white_matter_hyperintensities_segmen/config/FastSurferCNN_sagital.yaml --train_path /localmount/volume-ssd/users/uline/transfer/test_sagital_FlairT1.hdf5 --val_path /localmount/volume-ssd/users/uline/transfer/val_sagital_FlairT1.hdf5 --num_channels $num_c


echo -e "\e[1;37;41m Finish Script \e[0m" 
