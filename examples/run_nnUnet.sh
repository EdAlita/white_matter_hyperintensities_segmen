#! /bin/bash

echo -e "\e[1;37;41m Activating virtual env \e[0m"

source /home/uline/Desktop/nnUnet/.venv_nnunet/bin/activate

echo -e "\e[1;37;41m running flair \e[0m"

nnUNet_train 2d nnUNetTrainerV2_Fast 001 0 --npz

nnUNet_train 2d nnUNetTrainerV2_Fast 002 0 --npz

nnUNet_train 2d nnUNetTrainerV2_Fast 003 0 --npz

echo -e "\e[1;37;41m running t1 \e[0m"

nnUNet_train 2d nnUNetTrainerV2_Fast 004 0 --npz

nnUNet_train 2d nnUNetTrainerV2_Fast 005 0 --npz

nnUNet_train 2d nnUNetTrainerV2_Fast 006 0 --npz

echo -e "\e[1;37;41m running t2 \e[0m"

nnUNet_train 2d nnUNetTrainerV2_Fast 007 0 --npz

nnUNet_train 2d nnUNetTrainerV2_Fast 008 0 --npz

nnUNet_train 2d nnUNetTrainerV2_Fast 009 0 --npz

echo -e "\e[1;37;41m running t1flair \e[0m"

nnUNet_train 2d nnUNetTrainerV2_Fast 010 0 --npz

nnUNet_train 2d nnUNetTrainerV2_Fast 011 0 --npz

nnUNet_train 2d nnUNetTrainerV2_Fast 012 0 --npz

echo -e "\e[1;37;41m running t2flair \e[0m"

nnUNet_train 2d nnUNetTrainerV2_Fast 013 0 --npz

nnUNet_train 2d nnUNetTrainerV2_Fast 014 0 --npz

nnUNet_train 2d nnUNetTrainerV2_Fast 015 0 --npz

echo -e "\e[1;37;41m running flairt1t2 \e[0m"

nnUNet_train 2d nnUNetTrainerV2_Fast 016 0 --npz

nnUNet_train 2d nnUNetTrainerV2_Fast 017 0 --npz

nnUNet_train 2d nnUNetTrainerV2_Fast 018 0 --npz

deactivate

echo -e "\e[1;37;41m finishing script \e[0m"
