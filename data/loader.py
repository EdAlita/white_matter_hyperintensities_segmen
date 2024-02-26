from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from utils.stardarize_nii_files import map_size
from utils.preprocessing import min_max_normalization


class CustomDataset(Dataset):
    def __init__(self,
                 root_dir: Path,
                 transform: list = [],
                 wmh_split: str = 'train',
                 data_type: str = 'FLAIR',
                 data_slice: str = 'Axial',
                 verbose: bool = False,
                 plot_result: bool = False):

        path_csv = root_dir / 'wmh_overall.csv'
        data_frame = pd.read_csv(path_csv)

        self.wmh_split = wmh_split
        self.root_dir = root_dir
        self.transform = transform
        self.data_type = data_type
        self.data_slice = data_slice
        self.verbose = verbose
        self.plot_result = plot_result
        self.image_id = data_frame[data_frame.wmh_split == self.wmh_split].reset_index()

        supported_data_types = ['FLAIR', 'T1', 'FLAIRT1', 'T2']
        supported_data_slices = ['Axial', 'Sagital', 'Coronal']

        assert self.root_dir.is_dir() or self.root_dir.exists(), f'The provided directory does not exist: {self.root_dir}'
        assert self.data_type in supported_data_types, f'Invalid data type. Select from {supported_data_types}'
        assert self.data_slice in supported_data_slices, f'Invalid data type. Select from {supported_data_slices}'
        self.files_to_analyze = []

        for element in ['FLAIR', 'T1', 'T2']:
            if element in self.data_type:
                self.files_to_analyze.append(element)

    def __len__(self):
        return len(self.image_id)

    def __getitem__(self, idx):

        volume_dir = self.root_dir / self.image_id.iloc[idx, 1]

        if len(self.files_to_analyze) == 1:
            image_path = volume_dir / (self.files_to_analyze[0] + '.nii.gz')
            labels_path = volume_dir / 'lesion.nii.gz'

            volume_img = nib.load(image_path)
            lesion_img = nib.load(labels_path)

            if self.verbose:
                print(f'[info]: Loading {self.image_id.iloc[idx, 1]} with: ')
                print(f'   - From directory: {volume_dir}')
                print(
                    f'   - Image Size : {volume_img.header.get_data_shape()} and spacing: {volume_img.header.get_zooms()}')
                print(
                    f'   - Label Size : {lesion_img.header.get_data_shape()} and spacing: {lesion_img.header.get_zooms()}')

            if nib.aff2axcodes(volume_img.affine) != ('R', 'A', 'S'):
                volume_img = nib.as_closest_canonical(volume_img)
                labels_path = nib.as_closest_canonical(lesion_img)
                if self.verbose:
                    print(f'[info]: Converting {self.image_id.iloc[idx, 1]} to RAS orientation')

            volume_array = volume_img.get_fdata()
            labels_array = labels_path.get_fdata()

            if self.transform:
                if self.verbose: print(f'[info]: Transforming {self.image_id.iloc[idx, 0]} to {self.transform} space')
                volume_array = map_size(volume_array, self.transform, self.verbose)
                labels_array = map_size(labels_array, self.transform, self.verbose)

            volume_array = min_max_normalization(volume_array, max_val=256, name=self.files_to_analyze[0] + ' volume',
                                                 verbose=self.verbose)
            labels_array = min_max_normalization(labels_array, max_val=256, name=self.files_to_analyze[0] + ' label',
                                                 verbose=self.verbose)

            if self.plot_result:
                if self.verbose: print(f'[info]: Plotting')
                plt.subplot(1, 3, 1)
                plt.imshow(volume_array[:, :, 100], cmap='gray')
                plt.title('Axial')
                plt.axis('off')
                plt.subplot(1, 3, 2)
                plt.imshow(volume_array[:, 100, :], cmap='gray')
                plt.title('Coronal')
                plt.axis('off')
                plt.subplot(1, 3, 3)
                plt.imshow(volume_array[100, :, :], cmap='gray')
                plt.title('Sagital')
                plt.axis('off')
                plt.show()

            image_out = np.empty_like(volume_array)
            label_out = np.empty_like(labels_array)

            if self.verbose: print(f'[info]: Creating Slices in {self.data_slice}')

            if self.data_slice == 'Axial':
                for i in range(0, volume_array.shape[2]):
                    image_out[i, :, :] = volume_array[:, :, i]
                    label_out[i, :, :] = labels_array[:, :, i]
            elif self.data_slice == 'Coronal':
                for i in range(0, volume_array.shape[1]):
                    image_out[i, :, :] = volume_array[:, i, :]
                    label_out[i, :, :] = labels_array[:, i, :]
            elif self.data_slice == 'Sagital':
                for i in range(0, volume_array.shape[1]):
                    image_out[i, :, :] = volume_array[i, :, :]
                    label_out[i, :, :] = labels_array[i, :, :]

            return image_out, label_out


if __name__ == "__main__":
    data_set = CustomDataset(root_dir=Path('/localmount/volume-hd/users/uline/data_sets/MICCAI_2016/'),
                             transform=[256, 256, 256],
                             verbose=True,
                             #plot_result=True
                             )
    image, label = data_set.__getitem__(0)

    print(image.shape,label.shape)

    plt.subplot(1, 2, 1)
    plt.imshow(image[180, :, :], cmap='gray')
    plt.title('image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(label[180, :, :], cmap='gray')
    plt.title('label')
    plt.axis('off')

    plt.show()
