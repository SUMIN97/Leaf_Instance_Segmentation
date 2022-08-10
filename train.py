import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import time
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD
import torchvision
from tensorboardX import SummaryWriter

from models.wnet import WNet
from utils import *


class Trainer():
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device

        #Model
        self.model = WNet(cfg['unet_last_dim'], cfg['distance_feature_channels'], cfg['embedding_map_channels']).to(self.device)

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
                                                        drop_last=True,
                                                        num_workers=8)
        self.val_loader = torch.utils.data.DataLoader(IPELPlantDataset(cfg, mode='val'), batch_size=2)

        # DataParallel
        self.model = torch.nn.DataParallel(self.model).to(self.device)

        # Loss
        self.mse = torch.nn.MSELoss(reduction='sum')
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1)

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
            for idx, (rgb, label, gt_distmap, neighbor) in tqdm(enumerate(self.train_loader), total=nb):
            # for idx, (rgb, label, gt_distmap, neighbor) in enumerate(self.train_loader):
                glob_step = idx + nb * epoch

                rgb = rgb.to(self.device).float()
                label = label.to(self.device)
                gt_distmap = gt_distmap.to(self.device)

                pd_distmap, efeat = self.model(rgb)

                bs = efeat.shape[0]
                # Distance Map Loss
                # loss_distmap = self.mse(pd_distmap, gt_distmap) / bs
                loss_distmap = torch.square(pd_distmap - gt_distmap)
                loss_distmap = loss_distmap.mean()

                # Embedding Loss
                label_list = [label[i] for i in range(bs)]
                efeat_list = [efeat[i] for i in range(bs)]
                neighbor_list = [neighbor[i] for i in range(bs)]

                loss_intra, loss_inter = multi_apply(self.get_inter_and_inter_loss, label_list, efeat_list, neighbor_list)

                loss_intra = torch.stack(loss_intra).mean()
                loss_inter = torch.stack(loss_inter).mean()

                loss_distmap *= self.cfg.distmap_weight
                loss_intra *= self.cfg.intra_weight
                loss_inter *= self.cfg.inter_weight

                loss = loss_distmap + loss_intra + loss_inter

                #log
                if idx % 10 == 0:
                    print(f"loss distmap:{loss_distmap.item()}, intra:{loss_intra.item()}, inter:{loss_inter.item()}")

                self.writer.add_scalar("Train Loss/distmap", loss_distmap.item(), glob_step)
                self.writer.add_scalar("Train Loss/intra", loss_intra.item(), glob_step)
                self.writer.add_scalar("Train Loss/inter", loss_inter.item(), glob_step)
                self.writer.add_scalar("learning rate", self.scheduler.get_lr(), glob_step)

                if epoch > 5 and (idx % 50 == 0):
                    # rgb
                    img_grid = torchvision.utils.make_grid([rgb[0].cpu(), rgb[0].cpu()])
                    self.writer.add_image(f'Train RGB/{glob_step}', img_grid, glob_step)
                    # distmap
                    img_grid = torchvision.utils.make_grid([gt_distmap[0].cpu(), pd_distmap[0].cpu()])
                    self.writer.add_image(f'Train Distmap/{glob_step}', img_grid, glob_step)

                    #label
                    gt_label = label[0]
                    gt_label = get_colored_tensor(gt_label)

                    pd_seeds, pd_label = self.get_prediction(pd_distmap[0,0].cpu().detach().numpy(),
                                                    efeat[0].cpu().detach())

                    pd_seeds = torch.from_numpy(pd_seeds)

                    pd_label = get_colored_tensor(torch.from_numpy(pd_label))
                    img_grid = torchvision.utils.make_grid([gt_label, pd_label])
                    self.writer.add_image(f'Train Mask/{glob_step}', img_grid, glob_step)
                    self.writer.add_image(f'Train Prediction Seeds/{glob_step}', pd_seeds, glob_step, dataformats='HW')

                #Backward
                loss.backward()
                #Optimize
                self.optimizer.step()
                self.optimizer.zero_grad()

            #Validation
            self.model.eval()
            val_loss_distmap_sum, val_loss_intra_sum, val_loss_inter_sum = 0,0,0
            val_nb = len(self.val_loader)
            with torch.no_grad():
                for idx, (rgb, label, gt_distmap, neighbor) in tqdm(enumerate(self.val_loader), total=val_nb):
                    glob_step = idx + val_nb * epoch

                    rgb = rgb.to(self.device).float()
                    label = label.to(self.device)
                    gt_distmap = gt_distmap.to(self.device)

                    pd_distmap, efeat = self.model(rgb)

                    bs = efeat.shape[0]
                    # Distance Map Loss
                    # loss_distmap = self.mse(pd_distmap, gt_distmap) / bs
                    loss_distmap = torch.square(pd_distmap-gt_distmap)
                    loss_distmap = loss_distmap.mean()

                    # Embedding Loss

                    label_list = [label[i] for i in range(bs)]
                    efeat_list = [efeat[i] for i in range(bs)]
                    neighbor_list = [neighbor[i] for i in range(bs)]

                    loss_intra, loss_inter = multi_apply(self.get_inter_and_inter_loss, label_list, efeat_list,
                                                         neighbor_list)

                    loss_intra = torch.stack(loss_intra).mean()
                    loss_inter = torch.stack(loss_inter).mean()

                    val_loss_distmap_sum += loss_distmap.item()
                    val_loss_intra_sum += loss_intra.item()
                    val_loss_inter_sum += loss_inter.item()

                    if idx == 0 or idx == 3:
                        # rgb
                        img_grid = torchvision.utils.make_grid([rgb[0].cpu(), rgb[0].cpu()])
                        self.writer.add_image(f'Val RGB/{glob_step}', img_grid, glob_step)
                        # distmap
                        img_grid = torchvision.utils.make_grid([gt_distmap[0].cpu(), pd_distmap[0].cpu()])
                        self.writer.add_image(f'Val Distmap/{glob_step}', img_grid, glob_step)

                        # label
                        gt_label = label[0]
                        gt_label = get_colored_tensor(gt_label)

                        pd_seeds, pd_label = self.get_prediction(pd_distmap[0, 0].cpu().detach().numpy(),
                                                                 efeat[0].cpu().detach())

                        pd_seeds = torch.from_numpy(pd_seeds)

                        pd_label = get_colored_tensor(torch.from_numpy(pd_label))
                        img_grid = torchvision.utils.make_grid([gt_label, pd_label])
                        self.writer.add_image(f'Val Mask/{glob_step}', img_grid, glob_step)
                        self.writer.add_image(f'Val Prediction Seeds/{glob_step}', pd_seeds, glob_step, dataformats='HW')

                self.writer.add_scalar("Val Loss Sum/distmap", val_loss_distmap_sum, glob_step)
                self.writer.add_scalar("Val Loss Sum/intra", val_loss_intra_sum, glob_step)
                self.writer.add_scalar("Val Loss Sum/inter", val_loss_inter_sum, glob_step)

                ################################################################################
                #User Interactive
                # if epoch > 1 and idx %40 == 0:
                #     user_data = []
                #     pd_distmap = pd_distmap.squeeze().cpu().detach().numpy()
                #     efeat = efeat.cpu().detach()
                #     for i in range(bs):
                #         pd_seeds, pd_label = self.get_prediction(pd_distmap[i], efeat[i])
                #         pd_seeds = torch.from_numpy(pd_seeds)
                #         user_data.append(pd_seeds)
                #     user_data = torch.stack(user_data).unsqueeze(1)
                #
                #     pd_distmap, efeat = self.model(rgb, user_data=user_data)
                #
                #
                #     # Distance Map Loss
                #     loss_distmap = self.mse(pd_distmap, gt_distmap) / bs
                #
                #     # Embedding Loss
                #     label_list = [label[i] for i in range(bs)]
                #     efeat_list = [efeat[i] for i in range(bs)]
                #     neighbor_list = [neighbor[i] for i in range(bs)]
                #
                #     loss_intra, loss_inter = multi_apply(self.get_inter_and_inter_loss, label_list, efeat_list,
                #                                          neighbor_list)
                #
                #     loss_intra = torch.stack(loss_intra).mean()
                #     loss_inter = torch.stack(loss_inter).mean()
                #
                #     loss_distmap *= self.cfg.distmap_weight
                #     loss_intra *= self.cfg.intra_weight
                #     loss_inter *= self.cfg.inter_weight
                #
                #     loss = loss_distmap + loss_intra + loss_inter
                #
                #     print(f"After User Interactive loss distmap:{loss_distmap.item()}, intra:{loss_intra.item()}, inter:{loss_inter.item()}")
                #
                #     # Backward
                #     loss.backward()
                #     # Optimize
                #     self.optimizer.step()
                #     self.optimizer.zero_grad()


            #Scheduler
            self.optimizer.zero_grad()
            self.scheduler.step()
            print(f"\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.")

            if (epoch >0) and ((epoch +1) % 100 == 0):
                torch.save({
                    'WNet':self.model.state_dict()
                }, os.path.join(self.cfg.CHECKPOINTS_PATH, f'epoch{epoch}.pth'))

        self.writer.close()





    def get_inter_and_inter_loss(self, label, efeat, neighbor):
        n_ins = label.max() + 1 
        key_features = []

        #Key Feature
        c, h, w = efeat.shape
        efeat = efeat.permute(1,2,0).reshape(-1, c) #(h*w, c)


        for l in range(0, n_ins):
            #mean_feat
            fg = (label == l).reshape(-1)
            mean_feat = efeat[fg, :]
            mean_feat = torch.sum(mean_feat, dim=0) / fg.sum()
            key_features.append(mean_feat)

        #Intra Loss
        key_features = torch.stack(key_features, dim=0) #(n_ins, c)
        key_features = F.normalize(key_features, p=2, dim=1)

        # labelmap_flat = label.flatten().long()
        # key_features_expand = key_features[labelmap_flat]
        # loss_intra_a = 1 - self.cosine_similarity(key_features_expand, efeat)
        # loss_intra_a_m = loss_intra_a.mean()


        label = label.reshape(h*w, 1).expand(h*w, c)
        key_feat_each_pixel = torch.gather(key_features, dim=0, index=label) #(h*w, c)
        loss_intra = 1 - self.cosine_similarity(key_feat_each_pixel, efeat)
        loss_intra = loss_intra.mean()



        #Inter Loss
        key_feat_1 = key_features.unsqueeze(1).expand(n_ins, n_ins, c)
        key_feat_2 = key_feat_1.clone().permute(1,0,2)
        key_feat_1 = key_feat_1.reshape(-1, c)
        key_feat_2 = key_feat_2.reshape(-1, c)

        loss_inter = self.cosine_similarity(key_feat_1, key_feat_2).reshape(n_ins, n_ins).abs() #(n_key, n_key)

        neighbor_mask = torch.zeros(n_ins, n_ins, dtype=torch.float).to(self.device)
        for label_main in range(1, n_ins):
            for label_neighbor in neighbor[label_main -1]:
                if label_neighbor == 0: break

                neighbor_mask[label_main][label_neighbor] = 1.0

        neighbor_mask[0, :] = 1.
        neighbor_mask[:, 0] = 1.
        neighbor_mask[0,0] = 0.

        loss_inter = (loss_inter * neighbor_mask).sum(dim=1) / neighbor_mask.sum(dim=1)
        loss_inter = loss_inter.sum()

        return loss_intra, loss_inter

    def anchor_for_all_pixels(self, label, efeat, neighbor):
        n_ins = label.max() +1

        c, h, w = efeat.shape
        efeat = efeat.permute(1, 2, 0).view(-1, c)
        loss_positives, loss_negatives = 0.,0.


        for l in range(0, n_ins):
            # Positive
            fg_mask = (label == l).view(-1)
            n_fg = fg_mask.sum()
            anchor_feat = efeat[fg_mask, :]

            anchor = anchor_feat.unsqueeze(1).expand(n_fg, n_fg, c)
            positive = anchor.permute(1,0,2)
            loss_pos = self.cosine_similarity(anchor, positive).mean()
            loss_positives += loss_pos

            #Negative
            neg_mask = torch.zeros_like(label)
            for n_l in neighbor[l]:
                if n_l == 0: break
                neg_mask[label == n_l] = 1.
            n_neg = neg_mask.sum()
            neg_feat = efeat[neg_mask.view(-1), :]

            anchor = anchor_feat.unsqueeze(1).expand(n_fg,n_neg, c)
            negative = neg_feat.unsqueeze(0).expand(n_fg, n_neg, c)
            loss_neg = self.cosine_similarity(anchor, negative).mean()
            loss_negatives += loss_neg

        return loss_positives, loss_negatives








    def get_prediction(self, distmap, efeat):
        seeds = get_seeds(distmap, self.cfg.seeds_thres, self.cfg.seeds_min_dis)

        efeat = efeat.permute((1,2,0)) #(h,w,c)
        seg = mask_from_seeds(efeat, seeds, self.cfg.similarity_thres)
        seg = remove_noise(seg, distmap)

        return seeds, seg

    def get_prediction_v2(self, distmap, efeat):
        seeds = get_seeds(distmap, self.cfg.seeds_thres, self.cfg.seeds_min_dis)

        efeat = efeat.permute((1,2,0)) #(h,w,c)
        seg = mask_from_seeds_v2(efeat, seeds, self.cfg.similarity_thres, distmap)
        seg = remove_noise(seg, distmap)

        return seeds, seg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='configs/config.yml', help='yml file path')
    parser.add_argument('--pickle-path', type=str, default='./data/A1234_paths.pickle', help='data_paths file path')

    parser.add_argument('--exp-name', type=str, default='intra_weight_10', help='name for experiment folder')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--device', default='0,1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--resume-exp', type=str, default=None,
                        help='The prefix of the name of the experiment to be continued. '
                             'If you use this field, you must specify the "--resume-prefix" argument.')
    parser.add_argument('--workers', type=int, default=8, metavar='N', help='Dataloader threads.')
    parser.add_argument('--total_epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--img-size', type=int, default=512)

    args = parser.parse_args()

    return args



def main(args):
    cfg = init_experiment(args)
    trainer = Trainer(cfg)
    trainer.train()
    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args = parse_args()
    main(args)









