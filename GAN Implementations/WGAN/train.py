import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from model import Critic, Generator, initialize_weights
from tqdm import tqdm

# Hyperparameters etc.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEARNING_RATE = 5e-5  # could also use two lrs, one for gen and one for disc
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 1
NOISE_DIM = 100
NUM_EPOCHS = 5
FEATURES_DISC = 64
FEATURES_GEN = 64
CRITIC_ITERATIONS = 5
WEIGHT_CLIP = 0.01

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

if __name__ == "__main__":
    # comment mnist above and uncomment below if train on CelebA
    # dataset = datasets.ImageFolder(root="celeb_dataset", transform=transforms)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    gen = Generator(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
    critic = Critic(CHANNELS_IMG, FEATURES_DISC).to(device)
    initialize_weights(gen)
    initialize_weights(critic)

    opt_gen = optim.RMSprop(gen.parameters(), lr=LEARNING_RATE)
    opt_critic = optim.RMSprop(critic.parameters(), lr=LEARNING_RATE)

    fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)
    writer_real = SummaryWriter(f"logs/real")
    writer_fake = SummaryWriter(f"logs/fake")
    step = 0

    gen.train()
    critic.train()

    for epoch in range(NUM_EPOCHS):
        # Target labels not needed! <3 unsupervise  d
        for batch_idx, (real, _) in enumerate(tqdm(dataloader)):
            real = real.to(device)

            for _ in range(CRITIC_ITERATIONS):
                noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)
                fake = gen(noise)
                critic_real = critic(real).reshape(-1)
                critic_fake = critic(fake).reshape(-1)
                loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))
                writer_fake.add_scalar("loss_critic", loss_critic, global_step=step)
                critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()

                for p in critic.parameters():
                    p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

                output = critic(fake).reshape(-1)
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
                    fake = gen(fixed_noise)
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
