# WMHSegmentation-Thesis-Project

## Table of Contents

1. [**Introduction**](#introduction)
   - Overview and explanation of the project.
2. [**Objectives**](#objectives)
   - Development, automation, and validation of the lung image registration process.
3. [**Folders**](#folders)
   - Descriptions of various folders like Noteebooks, Data, Parameters, etc.
4. [**Installation and Usage**](#installation-and-usage)
   - Software requirements, installation guide, and usage instructions.
5. [**How to use it**](#wiki)
   - Links to instructons on how to use the model.
6. [**Authors**](#authors)
   - Contributions and profiles of the project team.
7. [**License**](#license)
   - Licensing information of the project.

## Introduction

This repository houses the project of the thesisAutomated Segmentation of White Matter Hyperintensities using Deep Learning , a crucial task in the early diagnosis and intervention of neurodegenerative diseases like Alzheimer’s. Our method uniquely integrates 3 architecture with multi-planar data representation and innovative training techniques to enhance segmentation accuracy and robustness.
Implemented models in this code: 

![alt text](https://github.com/EdAlita/white_matter_hyperintensities_segmen/blob/main/images/networks.png?raw=true)

## Objectives
1. Comprehensive analysis of medical image analysis for White Matter Hyperintensities (WMH).
2. Evaluate the performance of Dense UNet, FastSurferCNN, and nn-UNet.
3. Investigate the effects of multimodal information and varying input types and kernel sizes on these models.
4. Explore the impact of transfer learning.
5. Explore the models Generalizability.

## Folders

- [**Config**](config): configurations files for the networks and default configuration.
- [**Data**](data) : data scripts for ingesting, altering and preprocessing.
- [**Metrics**](metrics) : Evaluation Metrics use for the models.
- [**Models**](models) : Pices of code use in the creation of the models.
- [**Utils**](utils) : General codes use in the model.

## Installation and Usage

### Prerequisites

Ensure you have Python 3.8 or higher installed, along with PyTorch 1.7 and other necessary libraries detailed in the `requirements.txt`.


### Creating a Virtual Environment
To avoid conflicts with other Python projects, it's recommended to create a virtual environment:
1. Install `virtualenv` if you haven't already: `pip install virtualenv`
2. Create a new virtual environment: `virtualenv venv` (or `python -m venv venv` if using Python's built-in venv)
3. Activate the virtual environment:
   - On Windows: `venv\Scripts\activate`
   - On macOS and Linux: `source venv/bin/activate`
4. Your command prompt should now show the name of the activated environment.

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

## Wiki

How to used the scripts

- [**Pre-processing**](wiki/harmonize_data.md) 
- [**Generate Data**](wiki/generate_hdf5.md) 
- [**Train**](wiki/train.md)
- [**Inference**](wiki/inference.md) 

## Authors
- [Edwing Ulin](https://github.com/EdAlita)

## License
This project is licensed under the Creative Common Lincense - see the [LICENSE.md](LICENSE) file for details.
