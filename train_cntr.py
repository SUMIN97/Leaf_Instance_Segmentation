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
from utils import *


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

        #SyncBatchNorm
        # if cfg.sync_bn and cfg.device!= "cpu":
            # self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            # print('Using SyncBatchNorm')

        #Trainloader
        self.train_loader = torch.utils.data.DataLoader(PlantDataset(cfg, mode='train'),
                                                        batch_size=cfg.batch_size,
                                                        shuffle=True,
                                                        drop_last=True)
        self.val_loader = torch.utils.data.DataLoader(PlantDataset(cfg, mode='val'), batch_size=cfg.batch_size)

        #DataParallel
        self.model = torch.nn.DataParallel(self.model).to(self.device)

        #Loss
        self.mse = torch.nn.MSELoss(reduction='none')
        self.mse_mean = torch.nn.MSELoss(reduction='mean')
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=0)

        #Tensorboard log
        self.writer =  SummaryWriter(logdir=cfg.EXP_PATH)

    def train(self):
        nb = len(self.train_loader)
        start_epoch = 0

        for epoch in range(self.cfg.total_epochs):
            self.model.train()
            self.optimizer.zero_grad()

            #Start training
            t0 = time.time()
            for idx, (rgb, centersmap, center_points, distmap, binary_masks, is_instance, center_weights) in tqdm(enumerate(self.train_loader), total=nb):
                glob_step = idx + nb * epoch

                rgb = rgb.to(self.device).float()
                centersmap = centersmap.to(self.device)
                distmap = distmap.to(self.device)
                binary_masks = binary_masks.to(self.device)
                is_instance = is_instance.to(self.device)
                center_weights = center_weights.to(self.device)

                # pd_distmap, pd_centersmap, efeat = self.model(rgb)
                pd_centersmap, pd_distmap, efeat = self.model(rgb)
                # print('distmap max', distmap.max(), 'center max', centersmap.max())


                #Loss
                loss_distmap = self.mse_mean(pd_distmap, distmap)

                loss_centersmap = self.mse(pd_centersmap, centersmap) * center_weights #only fg
                loss_centersmap = loss_centersmap.sum() / center_weights.sum()

                loss_distmap *= 10
                loss_centersmap *= 10


                # center_points = [center_points[i] for i in range(self.cfg.batch_size)]
                # binary_masks = [binary_masks[i] for i in range(self.cfg.batch_size)]
                # is_instance = [is_instance[i] for i in range(self.cfg.batch_size)]
                # efeat = [efeat[i] for i in range(self.cfg.batch_size)]
                # loss_binarymask, pd_binary_masks = multi_apply(self.per_plant_loss, center_points,
                #                                       binary_masks, is_instance, efeat)
                # loss_binarymask = sum(loss_binarymask) / self.cfg.batch_size

                loss_binarymask, pred_binarymasks_for_vis = self.get_binarymask_loss(binary_masks, pd_centersmap, is_instance, efeat)
                loss_binarymask = loss_binarymask / is_instance.sum()

                loss = loss_centersmap  + loss_distmap + loss_binarymask

                if idx % 10 == 0:
                    # print('pd_distmap max', pd_distmap.max(), 'pd_centersmap max', pd_centersmap.max())
                    print(f"loss distmap:{loss_distmap.item()}, centersmap:{loss_centersmap.item()}, binarymask:{loss_binarymask.item()}")

                    self.writer.add_scalar("Train Loss/distmap", loss_distmap.item(), glob_step)
                    self.writer.add_scalar("Train Loss/centersmap", loss_centersmap.item(), glob_step)
                    self.writer.add_scalar("Train Loss/binarymasks", loss_binarymask.item(), glob_step)
                    self.writer.add_scalar("learning rate", self.scheduler.get_lr(), glob_step)


                if idx % 50 == 0:
                    #rgb
                    img_grid = torchvision.utils.make_grid([rgb[0].cpu(), rgb[0].cpu()])
                    self.writer.add_image(f'gt rgb/{glob_step}', img_grid, glob_step)

                    #distmap
                    img_grid = torchvision.utils.make_grid([distmap[0].cpu(), pd_distmap[0].cpu()])
                    self.writer.add_image(f'distmap gt and pd/{glob_step}', img_grid, glob_step)

                    #centermaps
                    img_grid = torchvision.utils.make_grid([centersmap[0].cpu(), pd_centersmap[0].cpu()])
                    self.writer.add_image(f'centermaps gt and pd/{glob_step}', img_grid, glob_step)

                    #label
                    label = torch.argmax(binary_masks[0], dim=0).cpu()
                    pd_label = torch.argmax(pred_binarymasks_for_vis[0], dim=0).cpu()

                    label = get_colored_tensor(label)
                    pd_label = get_colored_tensor(pd_label)
                    img_grid = torchvision.utils.make_grid([label, pd_label])
                    self.writer.add_image(f'label gt and pd/{glob_step}', img_grid, glob_step)

                #Backward
                loss.backward()

                #Optimize
                self.optimizer.step()
                self.optimizer.zero_grad()

            #Scheduler
            self.scheduler.step()
            print(f"\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.")

            if (epoch >0) and (epoch % 100 == 0):
                torch.save({
                    'WNet':self.model.state_dict()
                }, os.path.join(self.cfg.CHECKPOINTS_PATH, f'epoch{epoch}.pth'))

        self.writer.close()




    def per_plant_loss(self, centerpoints, pd_centersmap,  binarymasks, is_instance, efeat):
        """
        :param efeat: (c,h,w)
        :param centersmap: (1,h,w)
        :param centerpoints: (34,2)
        :param is_instance: (34)
        :param binary_masks: (34,h,w)
        :return:
        """
        n_centers = max(is_instance.sum(), 1)

        centersfeat = []

        for i in range(n_centers):
            # p = centerpoints[i]
            centersfeat.append(efeat[:,p[0],p[1]])
            centerpoints


        # intra_loss = multi_apply(self.leaf_intra_loss, centersfeat, binarymasks, efeat=efeat, reduction="sum")
        # intra_loss = torch.sum(intra_loss) / n_centers

        centersfeat = torch.stack(centersfeat, dim=0) #(n_p,c)
        pd_binarymasks = torch.einsum('pc,chw->phw', (centersfeat, efeat))
        binarymasks = binarymasks[is_instance, ...].float()
        mask_loss = self.sigmoid_focal_loss(pd_binarymasks, binarymasks,
                                            alpha=self.cfg.alpha, gamma=self.cfg.gamma,
                                            reduction="mean")
        # mask_loss = mask_loss.sum() / n_centers

        return mask_loss, pd_binarymasks

    def get_binarymask_loss(self, binarymasks, pd_centersmap, is_instance, efeat):
        '''
        :param binarymasks: (b,35,h,w)
        :param pd_centersmap: (b,1,h,w)
        :param is_instance:
        :param efeat:
        :return:
        '''
        pred_select = []
        b, c, h, w = efeat.shape
        k = self.cfg.weighted_num
        gt_num = is_instance.sum()

        guided_inst = pd_centersmap.expand_as(binarymasks) * binarymasks #(b,35,h,w)
        weighted_values, guided_index = torch.topk(guided_inst.reshape(*guided_inst.shape[:-2], -1), k=k, dim=-1) #(b,35,k)
        thing_num = int(max(is_instance.sum(dim=1).max(), 1))
        guided_index = guided_index[:, :thing_num, :]
        guided_index = guided_index.reshape(b, -1) #(b, thing_num*k)
        weighted_values = weighted_values[:, :thing_num, :] #(b,thing_num, k)
        is_instance = is_instance[:, :thing_num].reshape(-1)

        queryfeat = efeat.reshape(b, c, -1)
        guided_index =guided_index.unsqueeze(1).expand(-1,c,-1) #(b,c,thing_num*k)
        keyfeat = torch.gather(queryfeat, dim=2, index=guided_index) #(b,c,thing_num*k)
        # keyfeat = keyfeat.reshape(b,c,thing_num, self.cfg.weighted_num) #(b,c,thing_num,k)

        keyfeat = keyfeat.permute(0,2,1) #(b,thing_num*k, c)
        pred_binarymasks = torch.matmul(keyfeat, queryfeat) #(b, thing_num * k, h, w)
        pred_binarymasks = pred_binarymasks.reshape(b, thing_num, k, h, w)
        pred_binarymasks_for_vis = pred_binarymasks[:, :, 0, :, :]

        pred_binarymasks = pred_binarymasks.reshape(-1, k, h, w)[is_instance, ...]
        binarymasks = binarymasks[:, :thing_num, ...]
        binarymasks = binarymasks.unsqueeze(2).expand(b, thing_num, k, h, w)
        binarymasks = binarymasks.reshape(-1, k, h, w)[is_instance, ...]
        weighted_values = weighted_values.reshape(-1, k)[is_instance, ...]
        weighted_values = weighted_values / torch.clamp(weighted_values.sum(dim=-1, keepdim=True), min=1e-6)
        pred_binarymasks = torch.sigmoid(pred_binarymasks)


        binarymasks = binarymasks.reshape(-1, k, h*w)
        pred_binarymasks = pred_binarymasks.reshape(-1, k, h*w)

        #calculate dice loss
        loss_part = (pred_binarymasks **2).sum(dim=-1) + (binarymasks ** 2).sum(dim=-1)
        loss = 1 - 2 * (pred_binarymasks * binarymasks).sum(dim=-1) / torch.clamp(loss_part, min=1e-6)
        loss = loss * weighted_values

        return loss.sum(), pred_binarymasks_for_vis



    def leaf_intra_loss(self, centerfeat, binarymask, efeat, reduction="sum"):
        """
        :param centerfeat: c
        :param efeat: (c,h,w)
        :param reduction
        :return:
        """
        efeat = efeat[:, binarymask]
        centerfeat = centerfeat.expand_as(efeat)
        loss = self.cosine_similarity(centerfeat, efeat) #(h,w)
        if reduction == "sum":
            loss = loss.sum()
        elif reduction == "mean":
            loss = loss.mean()
        return loss

    def sigmoid_focal_loss(self, inputs, targets,
                           alpha: float = -1, gamma: float = 2,
                           reduction: str = "none"):
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                     classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            mode: A string used to indicte the optimization mode.
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor (1 - p_t) to
                   balance easy vs hard examples.
            reduction: 'none' | 'mean' | 'sum'
                     'none': No reduction will be applied to the output.
                     'mean': The output will be averaged.
                     'sum': The output will be summed.
        Returns:
            Loss tensor with the reduction option applied.
        """
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        return loss

    def weighted_dice_loss(self, prediction, target_seg):
        c, h, w = prediction.shape
        prediction = torch.sigmoid(prediction)
        prediction = prediction.reshape(-1, h*w)

        loss_part = (prediction ** 2).sum(dim=-1) + (target_seg **2).sum(dim=-1)





def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='exp')
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='configs/config.yml', help='yml file path')
    parser.add_argument('--pickle-path', type=str, default='./data/A1234_paths.pickle', help='data_paths file path')

    parser.add_argument('--exp-name', type=str, default='exp', help='name for experiment folder')
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



# See PyCharm help at https://www.jetbrains.com/help/pycharm/

