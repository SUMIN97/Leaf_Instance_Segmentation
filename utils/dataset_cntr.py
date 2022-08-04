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

from .augmentation import random_perspective, flipud, fliplr


class UsingCentermapPlantDataset(Dataset):
    def __init__(self, cfg, mode='train'):
        self.cfg = cfg
        self.mode = mode
        self.augmentation = cfg.augmentation
        self.load_data_list = cfg.load_data_list


        #data
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

        cfg  = self.cfg
        if self.augmentation:

            data = random_perspective(data)

            #Flip up-down
            if random.random() < cfg.flipud:
                data = flipud(data)

            #Flup left-right
            if random.random() < cfg.fliplr:
                data = fliplr(data)


        #change dtype
        # rgb = data['rgb'].transpose((2,0,1)) / 255.0
        rgb = torch.from_numpy(data['rgb'].transpose((2,0,1)) / 255.0)
        label = torch.from_numpy(data['label'].astype(np.int64))
        distmap = torch.from_numpy(data['distance_map'].astype(np.float32)) / 255.0 #/ 255.0 #0~1

        # get centers in range(0,1)
        centersmap = torch.zeros(cfg.img_size, cfg.img_size) #0~1
        # center_points = []
        # binary_masks = []
        center_points = torch.zeros(cfg.maximum_instance, 2, dtype=torch.int64)
        center_weights = torch.zeros(cfg.img_size, cfg.img_size, dtype=torch.float32)
        binary_masks = torch.zeros(cfg.maximum_instance, cfg.img_size, cfg.img_size, dtype=torch.uint8)
        is_instance = torch.zeros(cfg.maximum_instance, dtype=torch.bool)
        lbl_indexs = np.unique(label)


        for l in lbl_indexs:
            fg = (label == l)

            if l == 0: #bg
                pass
                # points = torch.nonzero(fg)
                # p = points[random.sample(range(len(points)), 1)]
                # center_points[l] = p

            else:
                hs, ws = torch.nonzero(fg, as_tuple=True)
                center_x, center_y = int(torch.mean(ws.float())), int(torch.mean(hs.float()))

                UsingCentermapPlantDataset.draw_gaussian(centersmap, (center_x, center_y), radius=cfg.center_radius,
                                           sigma_factor=cfg.center_sigma)
                center_points[l] = torch.tensor([center_y, center_x])
                center_weights[fg] = 1.

            binary_masks[l] = fg
            is_instance[l]=1


        # centersmap *= 255.0
        # centersmap[centersmap > 255.] = 255.

        # data = {
        #     'rgb':rgb,
        #     # 'label':label,
        #     'centersmap': centersmap,
        #     'center_points':center_points,
        #     'distmap':distance_map,
        #     'binary_masks':binary_masks,
        # }
        # return data
        return rgb, centersmap.unsqueeze(0), center_points, distmap.unsqueeze(0), binary_masks, is_instance, center_weights

    def __len__(self):
        return len(self.plants)

    def get_plant_name(self):
        return self.plants[self.index]

    def load_data(self, i):
        plant = self.plants[i]
        paths = self.data_paths[plant]
        input, neighbor = {}, ''
        for d in self.cfg.load_data_list:
            if d == 'neighbor':
                neighbor = np.load(paths[d])
            elif d == 'centers':
                continue
            else:
                input[d] = np.array(Image.open(paths[d]))

        #resize
        for d in self.load_data_list:
            if d == 'label':
                input[d] = cv2.resize(input[d], (self.cfg.img_size, self.cfg.img_size), interpolation=cv2.INTER_NEAREST)
            else:
                input[d] = cv2.resize(input[d], (self.cfg.img_size, self.cfg.img_size), interpolation=cv2.INTER_LINEAR)

        return input

    @staticmethod
    def gaussian2D(radius, sigma=1):
        m, n = radius
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        gauss = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        gauss[gauss < np.finfo(gauss.dtype).eps * gauss.max()] = 0
        return gauss

    @staticmethod
    def draw_gaussian(fmap, center, radius, k=1, sigma_factor=6, device=None):
        """
        Draw gaussian-based score map.
        """
        diameter = 2 * radius + 1
        gaussian = UsingCentermapPlantDataset.gaussian2D((radius, radius), sigma=diameter / sigma_factor)
        gaussian = torch.Tensor(gaussian)#.to(device=device)
        x, y = int(center[0]), int(center[1])
        height, width = fmap.shape[:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_fmap = fmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_fmap.shape) > 0:
            masked_fmap = torch.max(masked_fmap, masked_gaussian * k)
            fmap[y - top:y + bottom, x - left:x + right] = masked_fmap



if __name__ == '__main__':
    import yaml
    from easydict import EasyDict as edict
    import matplotlib.pyplot as plt

    with open('/configs/center_pred_config.yml', 'r') as f:
        cfg = edict(yaml.safe_load(f))

    train_ds = UsingCentermapPlantDataset(cfg)
    for (rgb, centers, center_points, distmap, binary_masks, is_instance) in train_ds:
        name = train_ds.get_plant_name()
        # rgb = data['rgb']
        # label = data['label']
        # centers = data['centersmap']
        # distance_map = data['distmap']


        # print(np.unique(label))
        # print((centers == 1).sum())
        # print(distance_map.shape)

        plt.imshow(rgb.numpy().transpose(1, 2, 0))
        plt.show()

        # plt.imshow(label.numpy())
        # plt.show()

        plt.imshow(centers.numpy())
        plt.show()

        plt.imshow(distmap.numpy())
        plt.show()
        # print(distance_map.max())

        # Image.fromarray(rgb.numpy().transpose(1, 2, 0)).save('rgb.png')
        centers = centers* 255
        Image.fromarray(centers.numpy().astype(np.uint8)).save(f'{name}_centers.png')

        break




