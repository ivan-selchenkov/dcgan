from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

from utils import display_data, scale

transform = transforms.ToTensor()

svhn_train = datasets.SVHN(root='data/', split='train', download=True, transform=transform)

batch_size = 128
num_workers = 0

train_loader = DataLoader(dataset=svhn_train, batch_size=batch_size, shuffle=True, num_workers=0)

display_data(train_loader)


