#!/bin/bash

dir=uk-biobank

cd $dir

echo -e "\e[1;37;41m Trainning CNN and UNet \e[0m"

list=("FlairT1")
num_c=14

for i in ${list[@]}
do	
	cd $i	
	
	echo -e "\e[1;37;41m $i run trainning\e[0m"
 
	echo -e "\e[1;37;41m Running UNET coronal training $i \e[0m"

	python3 /home/uline/Desktop/white_matter_hyperintensities_segmen/train-UKbiobank.py --config_path /home/uline/Desktop/white_matter_hyperintensities_segmen/config/FastSurferUNET_coronal.yaml --train_path /localmount/volume-ssd/users/uline/bio-bank-sample --val_path /localmount/volume-ssd/users/uline/ukbiobank/val_coronal_$i.hdf5 --num_channels $num_c
 
	echo -e "\e[1;37;41m Running CNN coronal training $i \e[0m"

	python3 /home/uline/Desktop/white_matter_hyperintensities_segmen/train-UKbiobank.py --config_path /home/uline/Desktop/white_matter_hyperintensities_segmen/config/FastSurferCNN_coronal.yaml --train_path /localmount/volume-ssd/users/uline/bio-bank-sample --val_path /localmount/volume-ssd/users/uline/ukbiobank/val_coronal_$i.hdf5 --num_channels $num_c

	cd ..

done

echo -e "\e[1;37;41m Finish Script \e[0m"
