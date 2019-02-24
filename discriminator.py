import torch.nn as nn
import torch.nn.functional as F


def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    layers = []

    # We don't use bias in case of batch normalization
    conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not batch_norm)

    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    return nn.Sequential(*layers)


class Discriminator(nn.Module):

    def __init__(self, conv_dim=32):
        super(Discriminator, self).__init__()

        self.conv_dim = conv_dim

        # 32x32 -> 16x16
        self.conv1 = conv(3, conv_dim, 4, batch_norm=False)
        # 16x16 -> 8x8
        self.conv2 = conv(conv_dim, conv_dim * 2, 4)
        # 8x8 -> 4x4
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4)

        # input 4x4 by conv_dim * 4 channels
        self.fc1 = nn.Linear(conv_dim * 4 * 4 * 4, 1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv2(x))

        x = x.view(-1, self.conv_dim * 4 * 4 * 4)

        x = self.fc1(x)

        # final sigmoid will be applied into BCEWithLogitsLoss function
        return x
