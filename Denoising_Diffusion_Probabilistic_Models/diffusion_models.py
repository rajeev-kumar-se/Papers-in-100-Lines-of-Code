# DDPM:
# 1. Initialize noise schedule
#     beta_t       = noise schedule
#     alpha_t      = 1 - beta_t
#     alpha_bar_t  = cumulative_product(alpha_t)
# 2. TRAINING:
#     a. Sample original images x0.
#     b. Randomly sample timestep t.
#     c. Generate Gaussian noise eps ~ N(0, I).
#     d. Create noisy image xt:
#        xt = sqrt(alpha_bar_t) * x0
#             + sqrt(1 - alpha_bar_t) * eps
#     e. Predict noise:
#        eps_pred = epsilon_theta(xt, t)
#     f. Calculate loss:
#        L = MSE(eps, eps_pred)
#     g. Update neural network:
#        optimizer.step()
# 3. SAMPLING:
#     a. Start with Gaussian noise:
#        xT ~ N(0, I)
#     b. For t = T to 1:
#        i.   Predict noise:
#             eps_pred = epsilon_theta(xt, t)
#        ii.  Calculate reverse mean:
#             mu = 1/sqrt(alpha_t) *
#                  (xt - (1-alpha_t)/sqrt(1-alpha_bar_t)
#                  * eps_pred)
#        iii. Sample random noise:
#             z ~ N(0, I), if t > 1
#             z = 0, if t = 1
#        iv.  Calculate previous image:
#             xt-1 = mu + sqrt(beta_t) * z
#     c. Return generated image x0.
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from keras.datasets.mnist import load_data
from unet import UNet
# declare once and use it everywhere
# torch.device = where tensor operations will be performed (CPU or GPU)
# torch.cuda.is_available() = only available in cuda.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data loading and preprocessing
# ([60000, 28, 28], [60000], [10000, 28, 28], [10000])
# trainX[0] = 28x28 image, trainy[0] = label of trainX[0]
# we should always normalize the data to [0, 1] range for better convergence
(trainX, trainy), (testX, testy) = load_data()
trainX = torch.tensor(trainX, dtype=torch.float32) / 255.
testX = torch.tensor(testX, dtype=torch.float32) / 255.
trainy = torch.as_tensor(trainy)
testy = torch.as_tensor(testy)

# sample batch 
def sample_batch(batch_size):
    # Generates a random permutation of integers from 0 to 59999.
    indices = torch.randperm(trainX.shape[0])[:batch_size]
    # [Batch, Height, Width] -> [Batch, Channel, Height, Width]
    data = trainX[indices].unsqueeze(1).to(device)
    # [64, 1, 28, 28] -> [64, 1, 32, 32], Why?
    return torch.nn.functional.interpolate(data, 32)

# main model class with init, sampling and training methods
class DiffusionModel(nn.Module):

    def __init__(self, T: int, model: nn.Module):
        super().__init__()

        self.T = T
        self.function_approximator = model

        # B1 = 1e-4, B2, ..., BT=2e-2
        beta = torch.linspace(1e-4, 0.02, T)
        alpha = 1. - beta
        # why dim=0?
        alpha_bar = torch.cumprod(alpha, dim=0)

        # parameters vs buffers?
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)

    def training(self, batch_size, optimizer):
        """
        Algorithm 1 in Denoising Diffusion Probabilistic Models
        """
        # [64, 1, 32, 32]
        x0 = sample_batch(batch_size)

        t = torch.randint(
            1, self.T + 1, (batch_size,),
            device=device,
            dtype=torch.long
        )
         # [64, 1, 32, 32]
        eps = torch.randn_like(x0)

        # Take one gradient descent step
        # alpha_bar_t = [64]
        # x0 = [64, 1, 32, 32]
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
            samples[i].squeeze(0).clip(0, 1).cpu(),
            cmap='gray'
        )

    plt.savefig(f'Imgs/samples.png')