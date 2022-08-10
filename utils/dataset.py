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
        data, neighbor = self.load_data(index)

        cfg = self.cfg
        if self.augmentation:
            if random.random() < cfg.random_perspective:
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
        rgb = self.input_normalize(rgb)
        label = torch.from_numpy(data['label'].astype(np.int64))
        distmap = torch.from_numpy(data['distance_map'].astype(np.float32)) / 255.0  # / 255.0 #0~1
        distmap = distmap.unsqueeze(0)
        neighbor = torch.from_numpy(neighbor)

        return rgb, label, distmap, neighbor

    def __len__(self):
        return len(self.plants)

    def load_data(self, i):
        plant = self.plants[i]
        paths = self.data_paths[plant]
        input, neighbor = {}, ''
        for d in self.cfg.load_data_list:
            if d == 'neighbor':
                neighbor = np.load(paths[d])
            elif d == 'centers':
                continue
            elif d == 'label':
                label = np.array(Image.open(paths[d]))
                input[d] = cv2.resize(label, (self.cfg.img_size, self.cfg.img_size), interpolation=cv2.INTER_NEAREST)
            elif d == 'rgb' or d == 'distance_map':
                data = np.array(Image.open(paths[d]))
                input[d] = cv2.resize(data, (self.cfg.img_size, self.cfg.img_size), interpolation=cv2.INTER_NEAREST)
            else:
                assert 'Error in load data'

        return input, neighbor

    def input_normalize(self,image):
        # total channel
        # image = torch.div(image - torch.mean(image), torch.std(image))

        # each channel
        orig_dtype = image.dtype
        image_mean = torch.mean(image, dim=(-1, -2, -3))
        stddev = torch.std(image, axis=(-1, -2, -3))
        num_pixels = torch.tensor(torch.numel(image), dtype=torch.float32)
        min_stddev = torch.rsqrt(num_pixels)
        adjusted_stddev = torch.max(stddev, min_stddev)
        # normalize image
        image -= image_mean
        image = torch.div(image, adjusted_stddev)
        # make sure that image output dtype  == input dtype
        assert image.dtype == orig_dtype

        return image