import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class RegNet(nn.Module):
    def __init__(self):
        super(RegNet, self).__init__()

        # Backbone
        self.layer1 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            #nn.Linear(128 * 16 * 16, 1024),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(64, 8)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
    
    def forward(self, x1, x2):
        #print(x1.shape, x2.shape)
        x1 = self.layer1(x1)
        x2 = self.layer1(x2)
        #print("after layer 1",x1.shape, x2.shape)
        x = self._cost_volume(x1, x2)
        #print("after cost volume",x.shape)
        x = self.avg_pool(x)
        B,C,H,W = x.shape
        x = x.view(B,C)
        x = self.fc(x)
        return x
    
    @staticmethod
    def _cost_volume(x1, x2):
        N, C, H, W = x1.shape
        x1 = F.normalize(x1)
        x2 = F.normalize(x2)
        x1 = x1.reshape(N, C, H*W)
        x2 = x2.reshape(N, C, H*W)
        cv = torch.bmm(x1.transpose(1, 2), x2)
        cv = cv.reshape(N, H*W, H, W) #.reshape(N, H*W, H, W) #cv.reshape(N, H*W//4, H*2, W*2) #LOOK HERE
        return cv
