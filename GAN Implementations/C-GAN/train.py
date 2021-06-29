import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Critic, Generator, initialize_weights
from utils import gradient_penalty, save_checkpoint, load_checkpoint
from tqdm import tqdm

# Hyperparameters etc.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEARNING_RATE = 1e-4  # could also use two lrs, one for gen and one for disc
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 1
NUM_CLASSES = 10
GEN_EMBEDDING = 100
NOISE_DIM = 100
NUM_EPOCHS = 5
FEATURES_DISC = 16
FEATURES_GEN = 16
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10

transforms = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        ),
    ]
)

# If you train on MNIST, remember to set channels_img to 1
dataset = datasets.MNIST(
    root="dataset/", train=True, transform=transforms, download=False
)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

if __name__ == "__main__":
    # comment mnist above and uncomment below if train on CelebA
    # dataset = datasets.ImageFolder(root="celeb_dataset", transform=transforms)

    gen = Generator(
        NOISE_DIM, CHANNELS_IMG, FEATURES_GEN, NUM_CLASSES, IMAGE_SIZE, GEN_EMBEDDING,
    ).to(device)
    critic = Critic(CHANNELS_IMG, FEATURES_DISC, NUM_CLASSES, IMAGE_SIZE).to(device)

    # initialize weights
    initialize_weights(gen)
    initialize_weights(critic)

    # intialize optimizers
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

    # for tensorboard img_grid per step
    fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)
    writer_real = SummaryWriter(f"logs/real")
    writer_fake = SummaryWriter(f"logs/fake")
    step = 0

    gen.train()
    critic.train()

    for epoch in range(NUM_EPOCHS):
        for batch_idx, (real, labels) in enumerate(tqdm(dataloader)):
            real = real.to(device)
            labels = labels.to(device)

            # Train Critic: max E[critic(real)] - E[critic(fake)]
            # equivalent to minimizing the negative of that
            for _ in range(CRITIC_ITERATIONS):
                noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)
                fake = gen(noise, labels)
                critic_real = critic(real, labels).reshape(-1)
                critic_fake = critic(fake, labels).reshape(-1)
                gp = gradient_penalty(critic, labels, real, fake, device=device)
                loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake))
                    + LAMBDA_GP * gp
                )
                writer_fake.add_scalar("loss_critic", loss_critic, global_step=step)

                critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()

                # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
                output = critic(fake, labels).reshape(-1)
                loss_gen = -torch.mean(output)
                writer_fake.add_scalar("loss_gen", loss_gen, global_step=step)
                gen.zero_grad()
                loss_gen.backward()
                opt_gen.step()

            # Print losses occasionally and print to tensorboard
            if batch_idx % 100 == 0:
                # print(
                #     f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
                #       Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
                # )
                with torch.no_grad():
                    fake = gen(noise, labels)
                    # take out (up to) 32 examples
                    img_grid_real = torchvision.utils.make_grid(
                        real[:32], normalize=True
                    )
                    img_grid_fake = torchvision.utils.make_grid(
                        fake[:32], normalize=True
                    )

                    writer_real.add_image("Real", img_grid_real, global_step=epoch)
                    writer_fake.add_image("Fake", img_grid_fake, global_step=epoch)

            step += 1
