# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%

from typing import *
from numpy.core.numeric import outer

import torch
from mpmath import mp
from torch import tensor
from src.models.common import Decoder, Encoder
from torch import Tensor, nn
from torch.distributions import Distribution, Independent, Normal
from torch.distributions.kl import kl_divergence
from src.data.mocap import MotionCaptureDataset


import optuna

class VariationalAutoencoder(nn.Module):
    """A Variational Autoencoder with
    * a Gaussian observation model `p(x|z)`
    * a Gaussian prior `p(z) = N(z | 0, I)`
    * a Gaussian posterior `q(z|x) = N(z | \mu(x), \sigma(x))`
    """

    def __init__(
        self, feature_dim, latent_dim, encoder_params=None, decoder_params=None
    ):

        super().__init__()

        self.feature_dim = feature_dim
        self.latent_dim = latent_dim

        if encoder_params is None:
            encoder_params = {}
        self.encoder = Decoder(self.feature_dim, 2 * latent_dim, **encoder_params)

        if decoder_params is None:
            decoder_params = {}
        self.decoder = Encoder(self.latent_dim, 2 * self.feature_dim, **decoder_params)

        self.register_buffer("prior_params", torch.zeros((1, 2 * latent_dim)))

        self.to(torch.double)

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

    def get_loss(self, batch, return_kl=False, beta=1.0):

        output = self(batch)
        px, pz, qz, z = [output[k] for k in ["px", "pz", "qz", "z"]]
        kl_term = kl_divergence(Independent(qz, 1), Independent(pz, 1))

        loss = -px.log_prob(batch).sum(-1) + beta * kl_term

        if not return_kl:
            return loss.mean()
        else:
            return loss.mean(), kl_term.mean()

    def get_ELBO_per_obs(self, batch, beta=1.0):
        output = self(batch)
        px, pz, qz, z = [output[k] for k in ["px", "pz", "qz", "z"]]
        kl_term = kl_divergence(Independent(qz, 1), Independent(pz, 1))

        elbo = px.log_prob(batch).sum(-1) + beta * kl_term

        return elbo 

    def log_likelihood(self, x, S = 10):
         # define the posterior q(z|x) / encode x into q(z|x)
        qz = self.posterior(x)
        # define the prior p(z)
        pz = self.prior(batch_size=x.size(0))

        # sample S samples from the posterior per data point x 
        z = qz.rsample([S]) # [S, batchsize, latentdim]

        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)

        # Calculating Monte Carlo Estimate of log likelihood 
        sum_log_lik = px.log_prob(x).sum(-1) + pz.log_prob(z).sum(-1) - qz.log_prob(z).sum(-1)
        log_lik = torch.zeros(x.shape[0])
        for i in range(x.shape[0]):
            tmp = mp.log(sum([mp.exp(t) for t in  sum_log_lik[:,i].detach().numpy()]) / S)
            log_lik[i] = float(tmp) 

        ave_log_lik = log_lik.mean()
        n_in_ave = x.shape[0]

        return {"log_like": log_lik, "average_log_like": ave_log_lik, "n": n_in_ave}

if __name__ == "__main__":
    dataset = MotionCaptureDataset("07", test=True)
    n_features = dataset.n_features
    X = dataset.X

    vae = VariationalAutoencoder(
    feature_dim=n_features,
    latent_dim=3,
    encoder_params={
        "activation_function" : "Tanh"
    },
    decoder_params={
        "activation_function" : "Tanh"
    },
)
    output = vae(X)
    #print(vae.get_loss(batch = X[0:5,:]))

    print(vae.log_likelihood(x = X, S = 1000))
    

