# WMHSegmentation-DenseUNet

This repository houses the Dense U-Net driven approach to automate the segmentation of White Matter Hyperintensities (WMH) from MRI data, a crucial task in the early diagnosis and intervention of neurodegenerative diseases like Alzheimer’s. Our method uniquely integrates Dense U-Net architecture with multi-planar data representation and innovative training techniques to enhance segmentation accuracy and robustness.

## Key Features

- **Dense U-Net Implementation:** Utilizes the Dense U-Net architecture for high-precision segmentation of WMH.
- **Multi-planar MRI Data Handling:** Adapts to the complex anatomical variations present in different patients using multi-planar data.
- **Robust Validation:** Extensively validated on datasets from the Rhineland Study and UK Biobank, demonstrating superior performance over traditional segmentation methods.
- **Open-Source Framework:** Available for the community to collaborate, further research, and adapt in clinical settings.

## Repository Contents

- **Source Code:** Complete implementation of the Dense U-Net model configured for various datasets.
- **Data Preprocessing Scripts:** Tools for preparing MRI data for training and evaluation.
- **Evaluation Scripts:** Scripts to quantitatively assess the model's performance using standard metrics.
- **Documentation:** Detailed guides and instructions on setting up, training, and deploying the segmentation models.

## Getting Started

### Prerequisites

Ensure you have Python 3.8 or higher installed, along with PyTorch 1.7 and other necessary libraries detailed in the `requirements.txt`.

### Installation

Clone the repository using:

```bash
git clone https://github.com/EdAlita/WMHSegmentation-DenseUNet.git
cd WMHSegmentation-DenseUNet
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

For detailed usage and additional parameters, refer to the documentation.
