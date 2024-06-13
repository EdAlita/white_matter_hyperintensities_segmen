#!/bin/bash

dir=Flair

echo -e "\e[1;37;41m $dir bio-evaluation\e[0m"

python3 /home/uline/Desktop/white_matter_hyperintensities_segmen/utils/create_2_D.py --out_path /localmount/volume-hd/users/uline/segmentation_results/miccai_2017/transfer_lr/test_set --npz_path /localmount/volume-hd/users/uline/segmentation_results/miccai_2017/transfer_lr/out

python3 /home/uline/Desktop/white_matter_hyperintensities_segmen/evaluatefrom_dir.py --eval_path /localmount/volume-hd/users/uline/segmentation_results/miccai_2017/transfer_lr/test_set

echo -e "\e[1;37;41m Finish Script \e[0m" 
