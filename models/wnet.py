import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet import UNet, DoubleConv, TrippleConv, UNet_with_two_decoder

class WNet(nn.Module):
    def __init__(self, unet_last_dim, distance_feature_channels, embedding_map_channels):
        super(WNet, self).__init__()

        self.input_dim = 3
        self.unet_ch = unet_last_dim
        self.dfeat_ch = distance_feature_channels
        self.emb_ch = embedding_map_channels

        self.dnet = UNet(self.input_dim)
        self.enet = UNet(self.dfeat_ch + self.input_dim)

        self.distmap_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(self.unet_ch, self.unet_ch, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.distmap_predictor = nn.Conv2d(64, 1, kernel_size=1)

        self.dfeat_head = nn.Sequential(
            nn.Conv2d(64, self.dfeat_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.dfeat_ch),
            nn.ReLU()
        )
        self.emb_predictor = nn.Conv2d(64, self.emb_ch, kernel_size=1)

        for predictor in [self.distmap_predictor, self.emb_predictor]:
            nn.init.normal_(predictor.weight, 0, 0.001)
            nn.init.constant_(predictor.bias, 0)


    def forward(self, input):
        x = self.dnet(input)

        distmap = self.distmap_head(x)
        distmap = self.distmap_predictor(distmap)

        dfeat = self.dfeat_head(x)
        dfeat = F.normalize(dfeat, p=2, dim=1)

        x = torch.cat((dfeat, input), dim=1)
        efeat = self.enet(x)
        efeat = self.emb_predictor(efeat)
        efeat = F.normalize(efeat, p=2, dim=1)

        return distmap, efeat




