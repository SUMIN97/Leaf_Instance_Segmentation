import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet import UNet, DoubleConv, TrippleConv, DropoutConv, UNet_with_two_decoder

class WNet(nn.Module):
    def __init__(self, distance_feature_channels, embedding_map_channels):
        super(WNet, self).__init__()

        self.input_dim = 3
        self.d_channels = distance_feature_channels
        self.e_channels = embedding_map_channels

        self.dnet = UNet(self.input_dim)
        self.enet = UNet(self.d_channels + self.input_dim)

        self.center_head = nn.Sequential(
            nn.Conv2d(64,64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.distmap_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dfeat_head = nn.Sequential(
            nn.Conv2d(64, self.d_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.d_channels),
            nn.ReLU()
        )

        self.embeddingconv = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Conv2d(64, self.e_channels, kernel_size=3, padding=1, bias=True),
            # nn.ReLU()
        )

        self.center_predictor = nn.Conv2d(64, 1, kernel_size=1)
        self.distmap_predictor = nn.Conv2d(64, 1, kernel_size=1)

        for predictor in [self.center_predictor, self.distmap_predictor]:
            nn.init.normal_(predictor.weight, 0, 0.001)
            nn.init.constant_(predictor.bias, 0)




    def forward(self, input):
        x = self.dnet(input)

        centers = self.center_head(x)
        distmap = self.distmap_head(x)
        dfeat = self.dfeat_head(x)

        centers = self.center_predictor(centers)
        distmap = self.distmap_predictor(distmap)


        x = torch.cat((dfeat, input), dim=1)
        efeat = self.enet(x)
        efeat = self.embeddingconv(efeat)
        # dfeat = F.normalize(dfeat, p=2, dim=1)
        # efeat = F.normalize(efeat, p=2, dim=1)

        return centers, distmap, efeat




