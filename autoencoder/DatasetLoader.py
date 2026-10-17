import base64
import random
from io import StringIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.Configuration import Configuration
class DatasetLoader:
    def __init__(self):
        self.conf = Configuration()
        self.loadDataset()
        #self.classes = len(self.dataset.classes)

    def loadDataset(self):
        # Define the transformations, including resizing to 512x512
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((512, 512)),  # Resize the images to 512x512
            transforms.ToTensor(),          # Convert images to PyTorch tensors
        ])

        def check_valid(path):
            #print(path,path[-4:] == '.png')
            if path[-4:] == '.png' or path[-4:] == '.jpg':
                try:
                    img = Image.open(path)
                    img.verify()
                    #print('Valid image')
                    return True
                except Exception:
                    return False

            return False
        print("Loading images and checking validity")
        # Load dataset using ImageFolder and apply the transformations
        self.dataset = ImageFolderwithFiles(root=self.conf.datasetfolder, transform=transform,is_valid_file=check_valid,allow_empty=True)

        # Create DataLoader
        self.dataloader = DataLoader(self.dataset, batch_size=40, shuffle=True, num_workers=4)

        # Example usage
        # for images, labels in self.dataloader:
        #     print(images.shape)  # Should be [32, 3, 512, 512] for a batch of 32 RGB images
        #     print(labels)  # Should be [32] for the batch of labels

class ImageFolderwithFiles(ImageFolder):

    def __init__(self, root,class_multipliers = 5, *args, **kwargs):
        """
        Args:
            root: Root directory path
            class_multipliers: Dict mapping class indices to number of times to duplicate
                             e.g., {0: 2, 3: 4} will duplicate class 0 items 2x and class 3 items 4x
            *args, **kwargs: Arguments for ImageFolder
        """
        super().__init__(root, *args, **kwargs)

        # Create expanded samples list with virtual duplicates
        original_samples = self.samples.copy()
        expanded_samples = []

        for path, class_idx in original_samples:
            # If this class should be oversampled, add multiple copies
            if not self.class_to_idx['unlabeled'] == class_idx:
                expanded_samples.extend([(path, class_idx)] * class_multipliers)
            else:
                # Otherwise just add the original sample once
                expanded_samples.append((path, class_idx))

        self.samples = expanded_samples
        # Update targets to match new samples list
        # print("self.targets before", self.targets)
        self.targets = [s[1] for s in self.samples]
        # print("self.targets after",self.targets )
        # exit()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index):

        sample, target = super(ImageFolderwithFiles,self).__getitem__(index)
        path, _ = self.samples[index]
        islabeled = not (self.class_to_idx['unlabeled'] == target)
        if islabeled:
            choice = random.randint(0,4)
            if choice == 0 or choice == 3 or choice == 4:
                # Inverti l'image
                sample = 1 - sample
            if choice == 1 or choice == 3:
                sample = torch.flip(sample,[1,2])
            if choice == 2 or choice == 4:
                sample = torch.flip(sample,[2,1])
        return sample, target, path, islabeled

    def shuffle_grayscale(self, image):
        # Get unique values and their shuffled version
        unique_values = torch.unique(image)
        shuffled_values = unique_values[torch.randperm(len(unique_values))]

        # Create new tensor for shuffled image
        shuffled_tensor = torch.zeros_like(image)

        # Map old values to new values
        for old_val, new_val in zip(unique_values, shuffled_values):
            mask = (image == old_val)
            shuffled_tensor[mask] = new_val
        return shuffled_tensor
