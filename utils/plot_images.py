import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import nibabel as nib
import numpy as np
from pathlib import Path
import argparse
from matplotlib.colors import ListedColormap
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure
from tqdm import tqdm


class NIfTIPlotter:
    def __init__(self, main_folder_path):
        self.main_folder_path = Path(main_folder_path)
        self.data_folder_path = Path('/localmount/volume-hd/users/uline/data_sets/CVD')
        self.th = 160

    def plot_nifti_folders(self):

        output_pdf_paths = [self.main_folder_path.parent / "sagital_results_nnUnet.pdf",
                            self.main_folder_path.parent / "axial_results_nnUnet.pdf",
                            self.main_folder_path.parent / "coronal_results_nnUnet.pdf"]
        for index, output_pdf_path in tqdm(enumerate(output_pdf_paths),desc='plotting planes... '):
            with PdfPages(output_pdf_path) as pdf:
                for subfolder in self.main_folder_path.iterdir():

                    file_paths = [f for f in sorted(subfolder.glob('*.nii.gz'))if f.is_file()]

                    n = len(file_paths)
                    cols = 3
                    rows = int(np.ceil(n / cols))
                    plt.figure(figsize=(cols * 5, rows * 5))
                    for i, file_path in enumerate(file_paths, start=1):
                        seg_img = nib.as_closest_canonical(nib.load(str(file_path))).get_fdata()
                        flai_img = nib.as_closest_canonical(nib.load(self.data_folder_path / f'{subfolder.name}' / 'FLAIR.nii.gz')).get_fdata()
                        lesion_img = nib.as_closest_canonical(nib.load(subfolder / 'lesion.nii.gz')).get_fdata()

                        lesion_img = binary_dilation(lesion_img, structure=generate_binary_structure(rank=3, connectivity=1), iterations=3).astype(lesion_img.dtype)
                        seg_img = np.logical_and(seg_img, lesion_img)

                        if index == 1:
                            seg_img = np.moveaxis(seg_img, [0, 1, 2], [1, 2, 0])
                            flai_img = np.moveaxis(flai_img, [0, 1, 2], [1, 2, 0])
                            lesion_img = np.moveaxis(lesion_img, [0, 1, 2], [1, 2, 0])
                        elif index == 2:
                            seg_img = np.moveaxis(seg_img, [0, 1, 2], [1, 0, 2])
                            flai_img = np.moveaxis(flai_img, [0, 1, 2], [1, 0, 2])
                            lesion_img = np.moveaxis(lesion_img, [0, 1, 2], [1, 0, 2])

                        seg_slice = seg_img[self.th, :, :]
                        flai_slice = flai_img[self.th,:, :]
                        lesion_slice = lesion_img[self.th,:, :]
                        plt.subplot(rows, cols, i)
                        plt.imshow(np.rot90(flai_slice), cmap='gray')
                        #if file_path.name != 'lesion.nii.gz':
                        #    red_cmap = ListedColormap(['none', "#2c7fb8"])
                        #   plt.imshow(np.rot90(lesion_slice), cmap=red_cmap)
                        red_cmap = ListedColormap(['none', "#a1dab4"])
                        plt.imshow(np.rot90(seg_slice), cmap=red_cmap)
                        plt.axis('off')
                        plt.title(file_path.stem)

                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot NIfTI files from subfolders into a single PDF.")
    parser.add_argument("main_folder_path", type=str,
                        help="Path to the main folder containing subfolders with NIfTI files.")

    args = parser.parse_args()

    plotter = NIfTIPlotter(args.main_folder_path)
    plotter.plot_nifti_folders()
