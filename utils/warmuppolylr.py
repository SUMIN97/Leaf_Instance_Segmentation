import torch
import math

"""
    Poly learning rate schedule used to train DeepLab.
    Paper: DeepLab: Semantic Image Segmentation with Deep Convolutional Nets,
        Atrous Convolution, and Fully Connected CRFs.
    Reference: https://github.com/tensorflow/models/blob/21b73d22f3ed05b650e85ac50849408dd36de32e/research/deeplab/utils/train_utils.py#L337  # noqa
    Refernece : https://gaussian37.github.io/dl-pytorch-lr_scheduler/
    """

class WarmupPolySchedule(torch.optim.lr_scheduler.LambdaLR):
    """
    Linear warmup and them poly decay.
    Linearly increases learning rate schedule from 0 to 1 until warmup_epoch.
    After warmup_epoch, lr polynomically decaies.
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, power, last_epoch=-1):

        def lr_lamda(epoch):
            lr_factor = 0.0
            if epoch < warmup_epochs:
                lr_factor =  float(epoch+1) / float(max(1.0, warmup_epochs))
            else:
                lr_factor = math.pow((1.0 - epoch / total_epochs), power)

            return lr_factor

        super(WarmupPolySchedule, self).__init__(optimizer, lr_lamda, last_epoch=last_epoch)

