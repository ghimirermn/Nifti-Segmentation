import os
import torch
import nibabel as nib
from torch.utils.data import Dataset
from skimage.transform import resize
import numpy as np

class NiftiDataset(Dataset):
    def __init__(self, image_folder, label_folder, resize_shape, normalize):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.resize_shape = resize_shape
        self.normalzie = normalize

        self.image_paths = sorted(os.listdir(image_folder))
        self.label_paths = sorted(os.listdir(label_folder))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_paths[idx])
        label_path = os.path.join(self.label_folder, self.label_paths[idx])

        image_name = self.image_paths[idx].split('.')[0]
        label_name = self.label_paths[idx].split('.')[0]

        # assert image_name == label_name, "Image and label names do not match."

        image = nib.load(image_path).get_fdata()
        label = nib.load(label_path).get_fdata()

        # image = np.moveaxis(image, -1, 0) #converting to channel-first numpy array
        # label = np.moveaxis(label, -1, 0) #converting to channel-first numpy array

        # Normalize the image if needed
        if self.normalzie is True:
            image = (image - image.min()) / (image.max() - image.min())

        # Resize the image and label
        if self.resize_shape is not None:
            image = self.resize_image(image)
            label = self.resize_image(label)

        label = self.binarize_label(label)

        # Convert to PyTorch tensors and add channel dimension
        image = torch.from_numpy(image).unsqueeze(0).float()
        label = torch.from_numpy(label).unsqueeze(0).float()

        return image, label

    def resize_image(self, image):
        resized_image = resize(image, (self.resize_shape, self.resize_shape, self.resize_shape), anti_aliasing=True)

        return resized_image

    def binarize_label(self, label):
        label[label >= 0.5] = 1
        label[label < 0.5] = 0
        return label

    def get_image_path(self, index):
        "used to return path for saving image during training"
        return os.path.join(self.image_folder, self.image_paths[index])

