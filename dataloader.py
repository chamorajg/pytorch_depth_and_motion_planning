import os
import json
import torch
import numpy as np
import pandas as pd

from PIL import Image
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

class DepthMotionDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, mode='train', transform=None,
                root_dir='./',):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_list = json.load(open('{}/data/{}.json'.format(root_dir, mode)))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(list(self.image_list.keys()))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_a, img_b = Image.open(self.image_list[str(idx)][0]), \
                            Image.open(self.image_list[str(idx)][1])
        if self.transform:
            sample_a = self.transform(img_a)
            sample_b = self.transform(img_b)
        return [sample_a, sample_b]