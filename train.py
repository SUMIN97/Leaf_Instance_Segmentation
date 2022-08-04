import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD
import torchvision
from tensorboardX import SummaryWriter

from models.wnet import WNet
from utils.dataset import IPELPlantDataset
from utils.warmuppolylr import WarmupPolySchedule

class Trainer():
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device

        #Model
        self.model = WNet(cfg['distance_feature_channels'], cfg['embedding_map_channels']).to(self.device)

        #Optimizer
        if cfg.adam:
            self.optimizer = Adam(self.model.parameters(), lr=float(cfg.lr), betas=(cfg.momentum, 0.999))
        else:
            self.optimizer = SGD(self.model.parameters(), momentum=cfg.momentum, nesterov=True)

        #Scheduler
        self.scheduler = WarmupPolySchedule(self.optimizer, warmup_epochs=cfg.warmup_epochs, total_epochs=cfg.total_epochs, power=cfg.power)
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[100,200,300,400], gamma=0.5)

        #Trainloader
        self.train_loader = torch.utils.data.DataLoader(IPELPlantDataset(cfg, mode='train'),
                                                        batch_size=cfg.batch_size,
                                                        shuffle=True,
                                                        drop_last=True)
        self.val_loader = torch.utils.data.DataLoader(IPELPlantDataset(cfg, mode='val'), batch_size=cfg.batch_size)

        # DataParallel
        self.model = torch.nn.DataParallel(self.model).to(self.device)

        # Loss
        self.mse = torch.nn.MSELoss(reduction='none')
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=0)

        # Tensorboard log
        self.writer = SummaryWriter(logdir=cfg.EXP_PATH)

    def train(self):
        nb = len(self.train_loader)
        start_epoch = 0

        for epoch in range(self.cfg.total_epochs):
            self.model.train()
            self.optimizer.zero_grad()

            # Start training
            t0 = time.time()
            for idx, (rgb, label, distmap) in tqdm(enumerate(self.train_loader), total=nb):
                glob_step = idx + nb * epoch

                rgb = rgb.to(self.device).float()
                label = label.to(self.device)
                distmap = distmap.to(self.device)


