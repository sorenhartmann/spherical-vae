# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from src.data.synthetic import SyntheticS2
from src.models.common import Decoder, Encoder, train_model
import numpy as np
import torch
from typing import *
from torch import nn, Tensor
from torch.distributions import Distribution, Normal, Independent
from torch.distributions.kl import kl_divergence

from torch.utils.data import random_split


class VariationalAutoencoder(nn.Module):
    """A Variational Autoencoder with
    * a Gaussian observation model `p(x|z)`
    * a Gaussian prior `p(z) = N(z | 0, I)`
    * a Gaussian posterior `q(z|x) = N(z | \mu(x), \sigma(x))`
    """

    def __init__(self, input_dimension, latent_features, output_dimension=None):

        super().__init__()

        self.input_dimension = input_dimension
        self.latent_features = latent_features
        self.observation_features = np.prod(input_dimension)

        if output_dimension is None:
            self.output_dimension = input_dimension
        else:
            self.output_dimension = output_dimension

        self.encoder = Encoder(self.observation_features, 2 * latent_features)
        self.decoder = Decoder(latent_features, 2 * self.observation_features)

        self.register_buffer(
            "prior_params", torch.zeros(torch.Size([1, 2 * latent_features]))
        )

    def posterior(self, x: Tensor) -> Distribution:
        """return the distribution `q(z|x) = N(z | \mu(x), \sigma(x))`"""
        # Compute the parameters of the posterior
        h_x = self.encoder(x)
        mu, log_sigma = h_x.chunk(2, dim=-1)

        # Return a distribution `q(z|x) = N(z | \mu(x), \sigma(x))`
        return Normal(mu, log_sigma.exp())

    def prior(self, batch_size: int = 1) -> Distribution:
        """return the distribution `p(z)`"""
        prior_params = self.prior_params.expand(
            batch_size, *self.prior_params.shape[-1:]
        )
        mu, log_sigma = prior_params.chunk(2, dim=-1)

        # return the distribution `p(z)`
        return Normal(mu, log_sigma.exp())

    def observation_model(self, z: Tensor) -> Distribution:
        """return the distribution `p(x|z)`"""
        h_z = self.decoder(z)
        mu, log_sigma = h_z.chunk(2, dim=-1)

        return Normal(mu, log_sigma.exp())

    def forward(self, x):
        # define the posterior q(z|x) / encode x into q(z|x)
        qz = self.posterior(x)

        # define the prior p(z)
        pz = self.prior(batch_size=x.size(0))

        # sample the posterior using the reparameterization trick: z ~ q(z | x)
        z = qz.rsample()

        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)

        return {"px": px, "pz": pz, "qz": qz, "z": z}

    def get_loss(self, batch):

        output = self(batch)
        px, pz, qz, z = [output[k] for k in ["px", "pz", "qz", "z"]]
        loss = -px.log_prob(batch).sum(-1) + kl_divergence(
            Independent(qz, 1), Independent(pz, 1)
        )
        return loss.mean()


if __name__ == "__main__":

    synthetic_s2 = SyntheticS2()
    vae = VariationalAutoencoder(synthetic_s2.n_features, 3)
    vae.to(torch.double)
    train_model(vae, synthetic_s2, checkpoint_path="models/synthetic/vae.pt")
