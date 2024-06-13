#!/bin/bash

folder=uk-biobank
dir=FlairT1
file=FLAIR
file2=T1
num=14
out_folder=out
cd $folder

cd $dir

mkdir $out_folder/CNN_axial

python3 /home/uline/Desktop/white_matter_hyperintensities_segmen/inference.py --dataset_path /localmount/volume-hd/users/uline/data_sets/CVD \
--volume_name $file.nii.gz \
--volume2_name $file2.nii.gz \
--volume3_name $file3.nii.gz \
--cfg_file /home/uline/Desktop/white_matter_hyperintensities_segmen/config/FastSurferCNN_axial.yaml \
--csv_file /localmount/volume-hd/users/uline/data_sets/CVD/wmh_overall.csv --plane axial --out_path $out_folder/CNN_axial \
--ckpt_path /localmount/volume-hd/users/uline/segmentation_results/aug_test/$dir/FastSurferCNN/checkpoints/FastSurferCNN_axial/Best_training_state.pkl \
--num_channels $num
 
mkdir $out_folder/CNN_coronal

python3 /home/uline/Desktop/white_matter_hyperintensities_segmen/inference.py --dataset_path /localmount/volume-hd/users/uline/data_sets/CVD \
--volume_name $file.nii.gz \
--volume2_name $file2.nii.gz \
--volume3_name $file3.nii.gz \
--cfg_file /home/uline/Desktop/white_matter_hyperintensities_segmen/config/FastSurferCNN_coronal.yaml \
--csv_file /localmount/volume-hd/users/uline/data_sets/CVD/wmh_overall.csv --plane coronal --out_path $out_folder/CNN_coronal \
--ckpt_path /localmount/volume-hd/users/uline/segmentation_results/aug_test/$dir/FastSurferCNN/checkpoints/FastSurferCNN_coronal/Best_training_state.pkl \
--num_channels $num

mkdir $out_folder/CNN_sagital

python3 /home/uline/Desktop/white_matter_hyperintensities_segmen/inference.py --dataset_path /localmount/volume-hd/users/uline/data_sets/CVD \
--volume_name $file.nii.gz \
--volume2_name $file2.nii.gz \
--volume3_name $file3.nii.gz \
--cfg_file /home/uline/Desktop/white_matter_hyperintensities_segmen/config/FastSurferCNN_sagital.yaml \
--csv_file /localmount/volume-hd/users/uline/data_sets/CVD/wmh_overall.csv --plane sagital --out_path $out_folder/CNN_sagital \
--ckpt_path /localmount/volume-hd/users/uline/segmentation_results/aug_test/$dir/FastSurferCNN/checkpoints/FastSurferCNN_sagital/Best_training_state.pkl \
--num_channels $num

mkdir $out_folder/UNET_axial

python3 /home/uline/Desktop/white_matter_hyperintensities_segmen/inference.py --dataset_path /localmount/volume-hd/users/uline/data_sets/CVD \
--volume_name $file.nii.gz \
--volume2_name $file2.nii.gz \
--volume3_name $file3.nii.gz \
--cfg_file /home/uline/Desktop/white_matter_hyperintensities_segmen/config/FastSurferUNET_axial.yaml \
--csv_file /localmount/volume-hd/users/uline/data_sets/CVD/wmh_overall.csv --plane axial --out_path $out_folder/UNET_axial \
--ckpt_path /localmount/volume-hd/users/uline/segmentation_results/aug_test/$dir/FastSurferUNET/checkpoints/FastSurferUNET_axial/Best_training_state.pkl \
--num_channels $num
 
mkdir $out_folder/UNET_coronal

python3 /home/uline/Desktop/white_matter_hyperintensities_segmen/inference.py --dataset_path /localmount/volume-hd/users/uline/data_sets/CVD \
--volume_name $file.nii.gz \
--volume2_name $file2.nii.gz \
--volume3_name $file3.nii.gz \
--cfg_file /home/uline/Desktop/white_matter_hyperintensities_segmen/config/FastSurferUNET_coronal.yaml \
--csv_file /localmount/volume-hd/users/uline/data_sets/CVD/wmh_overall.csv --plane coronal --out_path $out_folder/UNET_coronal \
--ckpt_path /localmount/volume-hd/users/uline/segmentation_results/aug_test/$dir/FastSurferUNET/checkpoints/FastSurferUNET_coronal/Best_training_state.pkl \
--num_channels $num

mkdir $out_folder/UNET_sagital

python3 /home/uline/Desktop/white_matter_hyperintensities_segmen/inference.py --dataset_path /localmount/volume-hd/users/uline/data_sets/CVD \
--volume_name $file.nii.gz \
--volume2_name $file2.nii.gz \
--volume3_name $file3.nii.gz \
--cfg_file /home/uline/Desktop/white_matter_hyperintensities_segmen/config/FastSurferUNET_sagital.yaml \
--csv_file /localmount/volume-hd/users/uline/data_sets/CVD/wmh_overall.csv --plane sagital --out_path $out_folder/UNET_sagital \
--ckpt_path /localmount/volume-hd/users/uline/segmentation_results/aug_test/$dir/FastSurferUNET/checkpoints/FastSurferUNET_sagital/Best_training_state.pkl \
--num_channels $num


