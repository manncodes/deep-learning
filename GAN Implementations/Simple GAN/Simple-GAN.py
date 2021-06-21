import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter


class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.gen(x)


# Hyperpamaters
# !GANs are incredibly sensitive to Hyperpameters
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 0.001
z_dim = 64
img_dim = 28 * 28 * 1
batch_size = 64
num_epochs = 10

disc = Discriminator(img_dim).to(device)
gen = Generator(z_dim, img_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5))]
)

dataset = datasets.MNIST(root="dataset/", transform=transforms, download=False)

loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
loss_fn = nn.BCELoss()
writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")

step = 0

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        # Disc-forward
        ### Train Discriminator: max log(D(real)) + log(1-D(G(z)))
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)  # reutilize

        disc_real = disc(real).view(-1)
        lossD_real = loss_fn(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).view(-1)  # to reutilize fake in generator do fake.detach
        lossD_fake = loss_fn(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2

        # Disc-backward
        disc.zero_grad()
        lossD.backward(retain_graph=True)  # retain_graph = True to reutilize `fake`

        # Disc-update
        opt_disc.step()

        ### Train Generator: min log(1-D(G(z))) <-> max log(D(G(z))) b'coz exp1 suffers from gradient saturation
        # Gen-forward
        output = disc(fake).view(-1)
        lossG = loss_fn(output, torch.ones_like(output))

        # Gen-backward
        gen.zero_grad()
        lossG.backward()

        # Gen-update
        opt_gen.step()

        print(
            f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
        )
        if batch_idx == 0:
            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real images", img_grid_real, global_step=step
                )

                step += 1
