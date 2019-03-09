import pickle as pkl

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

from discriminator import Discriminator, real_loss
from generator import Generator, fake_loss
from utils import display_data, scale, generate_z_vector, generate_plot
from torch import optim

transform = transforms.ToTensor()

svhn_train = datasets.SVHN(root='data/', split='train', download=True, transform=transform)

batch_size = 128
num_workers = 0

train_loader = DataLoader(dataset=svhn_train, batch_size=batch_size, shuffle=True, num_workers=0)

display_data(train_loader)

conv_size = 32
z_size = 100

D = Discriminator(conv_size)
G = Generator(z_size, conv_size)

cuda = False

if torch.cuda.is_available():
    cuda = True
    D = D.cuda()
    G = G.cuda()

lr = 0.0002
beta1 = 0.5
beta2 = 0.99

d_optim = optim.Adam(D.parameters(), lr, [beta1, beta2])
g_optim = optim.Adam(G.parameters(), lr, [beta1, beta2])


def train_discriminator(real_images, optimizer, batch_size, z_size):
    optimizer.zero_grad()

    if cuda:
        real_images = real_images.cuda()

    # Loss for real image
    d_real_loss = real_loss(D(real_images), cuda, smooth=True)

    # Loss for fake image
    fake_images = G(generate_z_vector(batch_size, z_size, cuda))
    d_fake_loss = fake_loss(D(fake_images), cuda)

    # add up loss and perform back-propagation
    d_loss = d_real_loss + d_fake_loss
    d_loss.backward()
    optimizer.step()

    return d_loss


def train_generator(optimizer, batch_size, z_size):
    optimizer.zero_grad()

    generated_images = G(generate_z_vector(batch_size, z_size, cuda))
    # training generator on real loss
    g_loss = real_loss(D(generated_images), cuda, smooth=True)
    g_loss.backward()
    optimizer.step()

    return g_loss

num_epochs = 30
samples = []
losses = []

print_every = 300

sample_size = 16
fixed_z = generate_z_vector(sample_size, z_size, cuda)

for epoch in range(num_epochs):
    for batch_i, (real_images, _) in enumerate(train_loader):
        batch_size = real_images.size(0)

        real_images = scale(real_images)

        d_loss = train_discriminator(real_images, d_optim, batch_size, z_size)
        g_loss = train_generator(g_optim, batch_size, z_size)

        # Print some loss stats
        if batch_i % print_every == 0:
            # print discriminator and generator loss
            print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                epoch + 1, num_epochs, d_loss.item(), g_loss.item()))

    losses.append((d_loss.item(), g_loss.item()))

    # generate and save sample, fake images
    G.eval()  # eval mode for generating samples
    samples_z = G(fixed_z)
    samples.append(samples_z)
    G.train()  # back to train mode

torch.save(D.state_dict(), './D.state')
torch.save(G.state_dict(), './G.state')

with open('train_samples.pkl', 'wb') as f:
    pkl.dump(samples, f)

generate_plot(losses)
