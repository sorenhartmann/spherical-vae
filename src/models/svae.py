from src.models.common import Decoder, Encoder, ModelParameterError
import torch
from torch.distributions import Distribution, Normal
from torch.distributions.kl import kl_divergence
from torch.nn import Module
from torch.tensor import Tensor
from src.distributions import VonMisesFisher, SphereUniform
from torch.utils.data import random_split
from scipy.special import ive


class SphericalVAE(Module):
    def __init__(
        self, feature_dim, latent_dim, encoder_params=None, decoder_params=None
    ):

        super().__init__()

        self.feature_dim = feature_dim
        self.latent_dim = latent_dim

        if encoder_params is None:
            encoder_params = {}
        self.encoder = Encoder(self.feature_dim, latent_dim + 1, **encoder_params)

        if decoder_params is None:
            decoder_params = {}
        self.decoder = Decoder(self.latent_dim, 2 * self.feature_dim, **decoder_params)

        self.to(torch.double)

    def posterior(self, x: Tensor) -> Distribution:
        """return the distribution `q(z|x) = vMF(z | \mu(x), \kappa(x))`"""
        # Compute the parameters of the posterior
        h_x = self.encoder(x)

        if h_x.isnan().any():
            raise ModelParameterError("NANs detected in encoder output")

        mu, log_k = h_x.split([self.latent_dim, 1], dim=-1)
        log_k = log_k.view(-1)
        mu = mu / mu.norm(dim=-1, keepdim=True)
        k = log_k.exp()

        if (k == 0).any():
            raise ModelParameterError("Extremely small values of k detected")

        # Return a distribution `q(z|x) = vMF(z | \mu(x), \kappa(x))`
        return VonMisesFisher(mu, k)

    def prior(self, batch_shape: torch.Size()) -> Distribution:
        """return the distribution `p(z)`"""
        # return the distribution `p(z)`
        return SphereUniform(dim=self.latent_dim - 1, batch_shape=batch_shape)

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
        z = qz.rsample(save_for_grad=True)

        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)

        return {"px": px, "pz": pz, "qz": qz, "z": z}

    def get_loss(self, batch, return_kl=False):

        output = self(batch)
        px, pz, qz, z = [output[k] for k in ["px", "pz", "qz", "z"]]
        kl_term = kl_divergence(qz, pz)
        loss = -px.log_prob(batch).sum(-1) + kl_term

        if not return_kl:
            return loss.mean()
        else:
            return loss.mean(), kl_term.mean()

    def fill_gradients_and_get_loss(self, x):

        output = self(x)
        px, pz, qz, z = [output[k] for k in ["px", "pz", "qz", "z"]]

        kl_term = kl_divergence(qz, pz)
        log_px = px.log_prob(x).sum(-1)
        loss = -log_px + kl_term
        loss = loss.mean()

        self.zero_grad()

        (log_px_d_k,) = torch.autograd.grad(log_px, qz.k, grad_outputs=torch.ones_like(qz.k),  retain_graph=True)
        (kl_term_d_k,) = torch.autograd.grad(kl_term, qz.k, grad_outputs=torch.ones_like(qz.k),  retain_graph=True)

        (loss_d_mu,) = torch.autograd.grad(loss, qz.mu, retain_graph=True)
        loss_d_decoder = torch.autograd.grad(
            loss, self.decoder.parameters(), retain_graph=True
        )

        eps = qz.saved_for_grad["eps"]
        w = qz.saved_for_grad["w"]
        b = qz.saved_for_grad["b"]

        corr_term = (
            w * qz.k
            + 1 / 2 * (qz.m - 3) * torch.log(1 - w ** 2)
            + torch.log(torch.abs(((-2 * b) / (((b - 1) * eps + 1) ** 2))))
        )

        (corr_term_d_k,) = torch.autograd.grad(
            corr_term, qz.k, grad_outputs=torch.ones_like(corr_term), retain_graph=True
        )
        with torch.no_grad():
            g_cor = log_px * (
                -ive(qz.m / 2, qz.k) / ive(qz.m / 2 - 1, qz.k) + corr_term_d_k
            )

        log_px_d_k_adj = log_px_d_k + g_cor
        loss_d_k = (-log_px_d_k_adj + kl_term_d_k) / len(qz.k)

        torch.autograd.backward(qz.k, grad_tensors=loss_d_k, retain_graph=True)
        torch.autograd.backward(qz.mu, grad_tensors=loss_d_mu, retain_graph=True)
        torch.autograd.backward(self.decoder.parameters(), grad_tensors=loss_d_decoder)

        return loss
