from torch import nn
import torch

criterion = nn.BCEWithLogitsLoss()

def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a transposed-convolutional layer, with optional batch normalization.
    """
    layers = []

    conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=not batch_norm)

    layers.append(conv)

    if (batch_norm):
        norm = nn.BatchNorm2d(out_channels)

        layers.append(norm)

    return nn.Sequential(*layers)


class Generator(nn.Module):
    def __init__(self, z_size, conv_dim=32):
        super(Generator, self).__init__()

        self.conv_dim = conv_dim

        self.fc = nn.Linear(z_size, conv_dim * 4 * 4 * 4)

        self.conv1 = deconv(conv_dim * 4, conv_dim * 2, 4)
        self.conv2 = deconv(conv_dim * 2, conv_dim, 4)
        self.conv3= deconv(conv_dim, 3, 4, batch_norm=False)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc(x)

        x = x.view(-1, self.conv_dim * 4, 4, 4) # batch_size, depth, 4x4

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.tanh(self.conv3(x))

        return x

def fake_loss(G_out: torch.Tensor, cuda):
    batch_size = G_out.size(0)
    labels = torch.ones(batch_size)

    if cuda:
        labels = labels.cuda()

    return criterion(G_out.squeeze(), labels)
