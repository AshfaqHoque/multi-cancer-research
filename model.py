# models.py
import torch
import torch.nn as nn
from torchvision.ops import DropBlock2d
from mamba_ssm import Mamba

from config import NUM_CLASSES, DROPBLOCK_P2, DROPBLOCK_BLOCKSIZE2, DROPBLOCK_P4, DROPBLOCK_BLOCKSIZE4


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class FeatureToSequence(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x shape: (batch, channels, height, width)
        batch, channels, h, w = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        return x, (h, w)


class SequenceToFeature(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, spatial_dims):
        batch, seq_len, channels = x.shape
        h, w = spatial_dims
        x = x.permute(0, 2, 1).reshape(batch, channels, h, w)
        return x


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.to_seq = FeatureToSequence()
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.to_feat = SequenceToFeature()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x_seq, spatial_dims = self.to_seq(x)
        x_seq = self.norm(x_seq)
        x_seq = self.mamba(x_seq)
        x = self.to_feat(x_seq, spatial_dims)
        return x


class CNNMambaHybrid(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CNNMambaHybrid, self).__init__()

        # Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Block 2 + CBAM + DropBlock
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.cbam2 = CBAM(64)
        self.dropblock2 = DropBlock2d(p=DROPBLOCK_P2, block_size=DROPBLOCK_BLOCKSIZE2)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.mamba1 = MambaBlock(d_model=64)

        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Block 4 + CBAM + DropBlock
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace=True)
        self.cbam4 = CBAM(256)
        self.dropblock4 = DropBlock2d(p=DROPBLOCK_P4, block_size=DROPBLOCK_BLOCKSIZE4)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.mamba2 = MambaBlock(d_model=256)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.output = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.dropblock2(self.cbam2(self.relu2(self.bn2(self.conv2(x))))))
        x = self.mamba1(x)

        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.dropblock4(self.cbam4(self.relu4(self.bn4(self.conv4(x))))))
        x = self.mamba2(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.output(x)
        return x