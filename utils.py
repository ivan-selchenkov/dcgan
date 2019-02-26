import unittest

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch import Tensor


def display_data(train_loader):
    data_iterator = iter(train_loader)
    images, labels = data_iterator.next()

    fig: Figure = plt.figure(figsize=(25, 4))

    plot_size = 20

    n_rows = 2
    n_cols = plot_size / 2

    for index in np.arange(plot_size):
        ax: Axes = fig.add_subplot(n_rows, n_cols, index + 1, xticks=[], yticks=[])
        ax.imshow(np.transpose(images[index], (1, 2, 0)))
        ax.set_title(str(labels[index].item()))

    plt.show()


def generate_z_vector(sample_size, z_size, cuda):
    z_vectors = np.random.uniform(-1, 1, size=(sample_size, z_size))
    z_vectors = torch.from_numpy(z_vectors).float()

    if cuda:
        z_vectors = z_vectors.cuda()

    return z_vectors


def generate_plot(losses):
    fig, ax = plt.subplots()
    losses = np.array(losses)
    plt.plot(losses.T[0], label='Discriminator')
    plt.plot(losses.T[1], label='Generator')
    plt.title("Training Losses")
    plt.legend()
    plt.show()


def scale(x: Tensor, feature_range=(-1, 1)):
    ''' Scale takes in an image x and returns that image, scaled
       with a feature_range of pixel values from -1 to 1.
       This function assumes that the input x is already scaled from 0-1.'''
    (range_min, range_max) = feature_range

    range_size = range_max - range_min
    range_shift = range_size / 2

    return (x - (range_shift - 0.5)) * range_size


class TestScale(unittest.TestCase):
    def test(self):
        test_scale_1 = scale(torch.tensor([0, 0.5, 1]), feature_range=(-1, 1))

        self.assertEqual(test_scale_1[0], -1)
        self.assertEqual(test_scale_1[1], 0)
        self.assertEqual(test_scale_1[2], 1)

        test_scale_2 = scale(torch.tensor([0, 0.5, 1]), feature_range=(0, 1))

        self.assertEqual(test_scale_2[0], 0)
        self.assertEqual(test_scale_2[1], 0.5)
        self.assertEqual(test_scale_2[2], 1)
