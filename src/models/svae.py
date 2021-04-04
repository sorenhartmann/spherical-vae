from sklearn.utils import validation
from src.data import synthetic
import torch
import numpy as np
from torch.distributions import Distribution, Normal
from torch.distributions.kl import kl_divergence
from torch.nn import Module, Linear, ReLU, Sequential
from torch.tensor import Tensor
from src.distributions import VonMisesFisher, SphereUniform, vmf_uniform_kl
from src.models.common import Encoder, Decoder
from torch.utils.data import random_split

from sklearn.model_selection import train_test_split

from src.data.genSynthData import genNoisySynthDataS2
from plotly import graph_objects as go

import random


class SphericalVAE(Module):

    def __init__(self, feature_dim, latent_dim):

        super().__init__()

        self.feature_dim = feature_dim
        self.latent_dim = latent_dim

        self.encoder = Encoder(feature_dim, latent_dim + 1)
        self.decoder = Decoder(latent_dim, 2 * self.feature_dim)

    def posterior(self, x: Tensor) -> Distribution:
        """return the distribution `q(z|x) = vMF(z | \mu(x), \kappa(x))`"""
        # Compute the parameters of the posterior
        h_x = self.encoder(x)
        mu, log_k = h_x.split([self.latent_dim, 1], dim=-1)
        log_k = log_k.squeeze()
        mu = mu / mu.norm(dim=-1, keepdim=True)

        # Return a distribution `q(z|x) = vMF(z | \mu(x), \kappa(x))`
        return VonMisesFisher(mu, log_k.exp())

    def prior(self, batch_shape: torch.Size()) -> Distribution:
        """return the distribution `p(z)`"""
        # return the distribution `p(z)`
        return SphereUniform(dim=self.latent_dim, batch_shape=batch_shape)

    def observation_model(self, z: Tensor) -> Distribution:
        """return the distribution `p(x|z)`"""
        h_z = self.decoder(z)
        mu, log_sigma = h_z.chunk(2, dim=-1)

        return Normal(mu, log_sigma.exp())

    def forward(self, x):
        # define the posterior q(z|x) / encode x into q(z|x)
        qz = self.posterior(x)

        # define the prior p(z)
        pz = self.prior(batch_shape=x.shape[:-1])

        # sample the posterior using the reparameterization trick: z ~ q(z | x)
        z = qz.rsample()

        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)

        return {"px": px, "pz": pz, "qz": qz, "z": z}


# %%
if __name__ == "__main__":

    torch.manual_seed(123)

    from src.data import SyntheticS2

    synthetic_s2 = SyntheticS2()

    train_size = int(0.8 * len(synthetic_s2))
    validation_size = len(synthetic_s2) - train_size

    train_dataset, validation_dataset = random_split(
        synthetic_s2, [train_size, validation_size]
    )


    ## Training loop
    n_epochs = 250
    batch_size = 32

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size)

    epoch_train_loss = [None] * len(train_loader)
    epoch_validation_loss = [None] * len(validation_loader)

    train_loss = [None] * n_epochs
    validation_loss = [None] * n_epochs

    # Model and optimizer
    feature_dim = synthetic_s2.n_features
    svae = SphericalVAE(feature_dim=feature_dim, latent_dim=3)
    svae.to(torch.double)

    optimizer = torch.optim.Adam(svae.parameters(), lr=1e-4)

    # with torch.autograd.detect_anomaly():

    for epoch in range(n_epochs):

        svae.train()

        for i, batch in enumerate(train_loader):
            # Forward pass
            output = svae.forward(batch)
            loss = -output["px"].log_prob(batch).sum(-1) + kl_divergence(
                output["qz"], output["pz"]
            )
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss[i] = loss.item()

        # Validating model
        svae.eval()

        with torch.no_grad():

            for i, batch in enumerate(validation_loader):
                # Forward pass
                output = svae.forward(batch)
                loss = -output["px"].log_prob(batch).sum(-1) + kl_divergence(
                    output["qz"], output["pz"]
                )
                loss = loss.mean()
                optimizer.zero_grad()

                epoch_validation_loss[i] = loss.item()

        train_loss[epoch] = sum(epoch_train_loss) / len(epoch_train_loss)
        validation_loss[epoch] = sum(epoch_validation_loss) / len(epoch_validation_loss)

        print(f"train loss: {train_loss[epoch]:.4f}")
        print(f"validation loss: {validation_loss[epoch]:.4f}")

    torch.save(svae.state_dict(), "models/svae.pt")


# %%
