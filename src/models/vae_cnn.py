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
    def __init__(self, image_size, latent_features):

        super().__init__()

        self.latent_features = latent_features

        self.encoder = Encoder(image_size, out_features = latent_features*2,  kernel_size = [3, 2], 
                               padding_size = [2, 1], out_channel_size = [7, 3], stride = [1,1],
                               activation_function = None, ffnn_layer_size = None,
                               dropout = None, dropout2d = None, maxpool = None)
    
        self.decoder = Decoder(in_features = latent_features,
                 reshape_features = self.encoder.last_im_size,
                 out_features = image_size, kernel_size = [3, 2], padding_size =  [2, 1],
                 out_channel_size = [7, 3], stride = [1,1],
                 activation_function = None, ffnn_layer_size = None,
                 dropout = None, dropout2d = None)
        
        self.register_buffer('prior_params', torch.zeros(torch.Size([1, 2*latent_features])))

    def posterior(self, x: Tensor) -> Distribution:
        """return the distribution `q(z|x) = N(z | \mu(x), \sigma(x))`"""
        # Compute the parameters of the posterior
        h_x = self.encoder(x)
        mu, log_sigma = h_x.chunk(2, dim=-1)

        # Return a distribution `q(z|x) = N(z | \mu(x), \sigma(x))`
        return Normal(mu, log_sigma.exp())
    
    def prior(self, batch_size: int=1) -> Distribution:
        """return the distribution `p(z)`"""
        prior_params = self.prior_params.expand(batch_size, *self.prior_params.shape[-1:])
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
    
        return {'px': px, 'pz': pz, 'qz': qz, 'z': z}

if __name__ == "__main__":
    
    torch.manual_seed(123)
    from src.data import SkinCancerDataset
    data = SkinCancerDataset(image_size=(225, 300))
    image_size = data.X.shape[1:]

    ham_vae = ConvolutionalVariationalAutoencoder(image_size=image_size, latent_features=3)

    ham_vae(data.X[:10,:,:,:])


