# Nifti-Segmentation

PyTorch-based 3D Vnet segmentation for medical imaging using Nifti dataset (.nii.gz) and leveraging the MONAI library.

## Dataset Format

The network assumes the following dataset format:

### Train
- Image
- Label

### Test
- Image
- Label

## Features

- [x] Parameter Logging
- [x] Tensorboard Logging
- [x] Test Outputs Saved at User Interval
- [x] Inference using model weight on User data 

## Description

Nifti-Segmentation is a powerful tool for performing 3D segmentation on medical imaging data. It is based on the Vnet architecture and built using PyTorch. The model works with the Nifti dataset format (.nii.gz) and takes advantage of the MONAI library for the VNet architecture as well as the Dice loss and metrics.

## How to Use

1. Clone the repository.
2. Prepare your Nifti dataset with the specified format.
3. Train the model using the provided training script.
4. Monitor the training progress with Tensorboard logging.
5. Evaluate the model on the test set and save the outputs at the desired intervals.
6. Visualize and analyze the segmentation results with ease.

## Acknowledgments

- The MONAI library for providing essential tools for medical imaging deep learning.
