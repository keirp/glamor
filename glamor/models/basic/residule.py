import torch
import torch.nn as nn


class BasicBlock(nn.Module):

    def __init__(self, channels, group_norm=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        if group_norm:
            self.bn1 = nn.GroupNorm(num_groups=4, num_channels=channels)
            self.bn2 = nn.GroupNorm(num_groups=4, num_channels=channels)
        else:
            self.bn1 = nn.BatchNorm2d(channels)
            self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += identity

        out = self.relu(out)

        return out


class ImpalaBlock(nn.Module):

    def __init__(self, in_channels, channels, group_norm=True):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=channels,
                              kernel_size=3, stride=1, padding=1, bias=False)
        self.pool = nn.MaxPool2d(3, stride=2)
        if group_norm:
            self.bn = nn.GroupNorm(num_groups=4, num_channels=channels)
        else:
            self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.block1 = BasicBlock(channels, group_norm=group_norm)
        self.block2 = BasicBlock(channels, group_norm=group_norm)

        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        out = self.conv(x)
        out = self.pool(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.block1(out)
        out = self.block2(out)
        return out


class Impala(nn.Module):

    def __init__(self, obs_shape, state_size, dropout_p=0, group_norm=True):
        super().__init__()

        self.obs_shape = obs_shape
        in_channels = self.obs_shape[0]

        self.block1 = ImpalaBlock(
            in_channels=in_channels, channels=16, group_norm=group_norm)
        self.block2 = ImpalaBlock(
            in_channels=16, channels=32, group_norm=group_norm)
        self.block3 = ImpalaBlock(
            in_channels=32, channels=32, group_norm=group_norm)

        test_input = torch.zeros(self.obs_shape).unsqueeze(0)
        out = self.block1(test_input)
        out = self.block2(out)
        out = self.block3(out)[0]

        self.out_size, self.out_shape = out.shape[0] * \
            out.shape[1] * out.shape[2], out.shape

        self.dropout = nn.Dropout(dropout_p)

        self.fc = nn.Linear(self.out_size, state_size)

    def forward(self, x):
        x = x.view(-1, *self.obs_shape)

        batch_size = x.shape[0]
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = out.view(batch_size, -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
