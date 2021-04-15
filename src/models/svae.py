from src.data.synthetic import SyntheticS2
import torch
from torch.distributions import Distribution, Normal
from torch.distributions.kl import kl_divergence
from torch.nn import Module
from torch.tensor import Tensor
from src.distributions import VonMisesFisher, SphereUniform
from src.models.common import Encoder, Decoder, train_model
from torch.utils.data import random_split


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

    def get_loss(self, batch):

        output = self(batch)
        px, pz, qz, z = [output[k] for k in ["px", "pz", "qz", "z"]]
        loss = -px.log_prob(batch).sum(-1) + kl_divergence(qz, pz)
        return loss.mean()


if __name__ == "__main__":

    synthetic_s2 = SyntheticS2()
    svae = SphericalVAE(synthetic_s2.n_features, 3)
    svae.to(torch.double)
    train_model(svae, synthetic_s2, checkpoint_path="models/synthetic/svae.pt")

# %%
