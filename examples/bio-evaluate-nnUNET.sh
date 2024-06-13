#!/bin/bash

list=("flair")

for i in ${list[@]}
do
	echo -e "\e[1;37;41m $i bio-evaluation\e[0m"

	python3 /home/uline/Desktop/white_matter_hyperintensities_segmen/utils/nnUnet-bio.py --out_path /localmount/volume-hd/users/uline/segmentation_results/miccai_2017/nnUNet/$i/test_set --npz_path /localmount/volume-hd/users/uline/segmentation_results/miccai_2017/nnUNet/$i/out

	python3 /home/uline/Desktop/white_matter_hyperintensities_segmen/evaluatefrom_dir.py --eval_path /localmount/volume-hd/users/uline/segmentation_results/miccai_2017/nnUNet/$i/test_set
done

echo -e "\e[1;37;41m Finish Script \e[0m" 
