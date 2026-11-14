import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from keras.datasets.mnist import load_data
from unet import UNet
# declare once and use it everywhere
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data loading and preprocessing
(trainX, trainy), (testX, testy) = load_data()
trainX = np.float32(trainX) / 255.
testX = np.float32(testX) / 255.

# sample batch 
def sample_batch(batch_size):
    indices = torch.randperm(trainX.shape[0])[:batch_size]
    data = torch.from_numpy(trainX[indices]).unsqueeze(1).to(device)
    return torch.nn.functional.interpolate(data, 32)

# main model class with init, sampling and training methods
class DiffusionModel(nn.Module):

    def __init__(self, T: int, model: nn.Module):
        super().__init__()

        self.T = T
        self.function_approximator = model

        beta = torch.linspace(1e-4, 0.02, T)
        alpha = 1. - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)

    def training(self, batch_size, optimizer):
        """
        Algorithm 1 in Denoising Diffusion Probabilistic Models
        """

        x0 = sample_batch(batch_size)

        t = torch.randint(
            1, self.T + 1, (batch_size,),
            device=device,
            dtype=torch.long
        )

        eps = torch.randn_like(x0)

        # Take one gradient descent step
        alpha_bar_t = self.alpha_bar[t - 1].unsqueeze(-1).unsqueeze(
            -1).unsqueeze(-1)

        eps_predicted = self.function_approximator(
            torch.sqrt(alpha_bar_t) * x0 +
            torch.sqrt(1 - alpha_bar_t) * eps,
            t - 1
        )

        loss = nn.functional.mse_loss(eps, eps_predicted)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    @torch.no_grad()
    def sampling(self, n_samples=1, image_channels=1, img_size=(32, 32)):
        """
        Algorithm 2 in Denoising Diffusion Probabilistic Models
        """

        x = torch.randn(
            (n_samples, image_channels, img_size[0], img_size[1]),
            device=device
        )

        for t in tqdm(range(self.T, 0, -1)):
            z = torch.randn_like(x) if t > 1 else torch.zeros_like(x)

            t = torch.ones(
                n_samples,
                dtype=torch.long,
                device=device
            ) * t

            beta_t = self.beta[t - 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            alpha_t = self.alpha[t - 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            alpha_bar_t = self.alpha_bar[t - 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            mean = 1 / torch.sqrt(alpha_t) * (
                x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) *
                self.function_approximator(x, t - 1)
            )

            sigma = torch.sqrt(beta_t)
            x = mean + sigma * z

        return x


if __name__ == "__main__":

    # Arch Prep
    batch_size = 64

    denoiser_model = UNet()
    diffusion_model = DiffusionModel(1000, denoiser_model).to(device)

    # Training
    optimizer = torch.optim.Adam(denoiser_model.parameters(), lr=2e-5)
    for epoch in tqdm(range(40_000)):
        loss = diffusion_model.training(batch_size, optimizer)

    # Inference
    nb_images = 81

    samples = diffusion_model.sampling(n_samples=nb_images)

    plt.figure(figsize=(17, 17))

    for i in range(nb_images):
        plt.subplot(9, 9, 1 + i)
        plt.axis('off')
        plt.imshow(
            samples[i].squeeze(0).clip(0, 1).data.cpu().numpy(),
            cmap='gray'
        )

    plt.savefig(f'Imgs/samples.png')