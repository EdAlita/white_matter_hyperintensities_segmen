#!/bin/bash

cd data
folder=combine
name=CVD+MICCAI2008

python3 /home/uline/Desktop/white_matter_hyperintensities_segmen/data/generate_hdf5.py --dataset_name train_axial_$name.hdf5 \
--dataset_path /localmount/volume-hd/users/uline/data_sets/$folder \
--volumen2_name T2.nii.gz \
--csv_file  /localmount/volume-hd/users/uline/data_sets/$folder/wmh_overall.csv \
--plane axial \
--datatype train

python3 /home/uline/Desktop/white_matter_hyperintensities_segmen/data/generate_hdf5.py --dataset_name val_axial_$name.hdf5 \
--dataset_path /localmount/volume-hd/users/uline/data_sets/$folder \
--volumen2_name T2.nii.gz \
--csv_file  /localmount/volume-hd/users/uline/data_sets/$folder/wmh_overall.csv \
--plane axial \
--datatype val

python3 /home/uline/Desktop/white_matter_hyperintensities_segmen/data/generate_hdf5.py --dataset_name train_coronal_$name.hdf5 \
--dataset_path /localmount/volume-hd/users/uline/data_sets/$folder \
--volumen2_name T2.nii.gz \
--csv_file  /localmount/volume-hd/users/uline/data_sets/$folder/wmh_overall.csv \
--plane coronal \
--datatype train

python3 /home/uline/Desktop/white_matter_hyperintensities_segmen/data/generate_hdf5.py --dataset_name val_coronal_$name.hdf5 \
--dataset_path /localmount/volume-hd/users/uline/data_sets/$folder \
--volumen2_name T2.nii.gz \
--csv_file  /localmount/volume-hd/users/uline/data_sets/$folder/wmh_overall.csv \
--plane coronal \
--datatype val

python3 /home/uline/Desktop/white_matter_hyperintensities_segmen/data/generate_hdf5.py --dataset_name train_sagital_$name.hdf5 \
--dataset_path /localmount/volume-hd/users/uline/data_sets/$folder \
--volumen2_name T2.nii.gz \
--csv_file  /localmount/volume-hd/users/uline/data_sets/$folder/wmh_overall.csv \
--plane sagital \
--datatype train

python3 /home/uline/Desktop/white_matter_hyperintensities_segmen/data/generate_hdf5.py --dataset_name val_sagital_$name.hdf5 \
--dataset_path /localmount/volume-hd/users/uline/data_sets/$folder \
--volumen2_name T2.nii.gz \
--csv_file  /localmount/volume-hd/users/uline/data_sets/$folder/wmh_overall.csv \
--plane sagital \
--datatype val


