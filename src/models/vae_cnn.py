# ordinary VAE for the HAM data
from src.models.cnn_encoder_decoder import Decoder, Encoder
import numpy as np
import torch
from typing import *
from torch import nn, Tensor
from torch.distributions import Distribution, Normal, Independent
from torch.distributions.kl import kl_divergence

from torch.utils.data import random_split


class ConvolutionalVariationalAutoencoder(nn.Module):
    """A Variational Autoencoder with
    * a Gaussian observation model `p(x|z)`
    * a Gaussian prior `p(z) = N(z | 0, I)`
    * a Gaussian posterior `q(z|x) = N(z | \mu(x), \sigma(x))`
    """

    def __init__(
        self, image_size, latent_dim, encoder_params=None, decoder_params=None
    ):

        super().__init__()

        self.latent_dim = latent_dim
        channels, height, width = image_size

        if encoder_params is None:
            encoder_params = {}

        self.encoder = Encoder(
            image_size=image_size, out_features=latent_dim * 2, **encoder_params
        )

        if decoder_params is None:
            decoder_params = {}

        self.decoder = Decoder(
            in_features=latent_dim,
            image_size=(2 * channels, height, width),
            **decoder_params
        )

        self.register_buffer(
            "prior_params", torch.zeros(torch.Size([1, 2 * latent_dim]))
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
        mu, log_sigma = h_z.chunk(2, dim=1)

        return Independent(Normal(mu, log_sigma.exp()), 3)

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

    def get_loss(self, x, return_kl=False, beta=1.0):
        output = self(x)
        px, pz, qz, z = [output[k] for k in ["px", "pz", "qz", "z"]]
        kl_term = kl_divergence(Independent(qz, 1), Independent(pz, 1))

        loss = -px.log_prob(x) + beta * kl_term

        if not return_kl:
            return loss.mean()
        else:
            return loss.mean(), kl_term.mean()


if __name__ == "__main__":

    torch.manual_seed(123)
    from src.data import SkinCancerDataset

    data = SkinCancerDataset(image_size=(225, 300))
    image_size = data.X.shape[1:]

    ham_vae = ConvolutionalVariationalAutoencoder(
        image_size=image_size, latent_dim=3
    )

    with torch.autograd.profiler.profile(record_shapes=True) as prof:
        loss = ham_vae.get_loss(data.X[:10, :, :, :])

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    
    
    
    # print(loss)
    # loss.backward()
    # with torch.no_grad():
    #    loss = ham_vae.get_loss(data.X[:10,:,:,:])
    #    print(loss)
