import cv2
import os
import numpy as np
import pickle
from PIL import Image
import random

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
#
from .augmentation import random_perspective, flipud, fliplr


class IPELPlantDataset(Dataset):
    def __init__(self, cfg, mode='train'):
        self.cfg = cfg
        self.mode = mode
        self.augmentation = cfg.augmentation
        self.load_data_list = cfg.load_data_list

        # data
        pickle_path = ''
        if mode == 'train':
            pickle_path = os.path.join(cfg.data_path, 'train.pickle')
        elif mode == 'val':
            pickle_path = os.path.join(cfg.data_path, 'val.pickle')
            self.augmentation = False
        else:
            assert "Error  in PlantDataset"

        with open(pickle_path, 'rb') as f:
            self.data_paths = pickle.load(f)
        self.plants = list(self.data_paths.keys())
        self.index = 0

    def __getitem__(self, index):
        self.index = index
        data = self.load_data(index)

        cfg = self.cfg
        if self.augmentation:

            data = random_perspective(data)

            # Flip up-down
            if random.random() < cfg.flipud:
                data = flipud(data)

            # Flup left-right
            if random.random() < cfg.fliplr:
                data = fliplr(data)

        # change dtype
        # rgb = data['rgb'].transpose((2,0,1)) / 255.0
        rgb = torch.from_numpy(data['rgb'].transpose((2, 0, 1)) / 255.0)
        label = torch.from_numpy(data['label'].astype(np.int64))
        distmap = torch.from_numpy(data['distance_map'].astype(np.float32)) / 255.0  # / 255.0 #0~1
        distmap = distmap.unsqueeze(0)

        return rgb, label, distmap