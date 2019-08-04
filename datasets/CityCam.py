import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class CityCam(Dataset):
    """
    Vehicle counting dataset with bounding box and vehicle density map

    args:
        root_dir: the root directory of CityCam, which should be 'CityCam/'
        dataset_type: 'Train' for the training set, 'Test' for the test set
        transform: torchvision.transforms
    """
    def __init__(self, root_dir, dataset_type='Train', transform=None):
        """

        """
        self.root_dir = root_dir
        self.transform = transform
        self.sample_list = []

        train_test_split_dir = os.path.join(self.root_dir, 'train_test_separation')
        downtown_path = os.path.join(train_test_split_dir, 'Downtown_' + dataset_type + '.txt')
        parkway_path = os.path.join(train_test_split_dir, 'Parkway_' + dataset_type + '.txt')

        frame_dirs = []
        with open(downtown_path) as f:
            frame_dirs.extend(f.readlines())

        with open(parkway_path) as f:
            frame_dirs.extend(f.readlines())

        for fd in frame_dirs:
            camera_dir = os.path(root_dir, fd.split('-')[0])
            frame_dir = os.path(camera_dir, fd)
            self.sample_list.extend(sorted(filter(lambda frame_name: frame_name.split('.')[-1] == 'jpg', os.listdir(frame_dir))))
        

    def __len__(self):
        return len(self.sample_list)


    def __getitem__(self, idx):
        pass