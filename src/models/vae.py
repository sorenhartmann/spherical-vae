# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from src.models.common import Decoder, Encoder
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
    def __init__(self, input_dimension, latent_features, output_dimension = None):

        super().__init__()

        self.input_dimension = input_dimension
        self.latent_features = latent_features
        self.observation_features = np.prod(input_dimension)

        if output_dimension is None:
            self.output_dimension = input_dimension
        else:
            self.output_dimension = output_dimension
        
        self.encoder = Encoder(self.observation_features, 2*latent_features)
        self.decoder = Decoder(latent_features, 2*self.observation_features)
        
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

    from src.data.synthetic import SyntheticS2

    dataset = SyntheticS2()

    train_size = int(0.8 * len(dataset))
    validation_size = len(dataset) - train_size

    train_dataset, validation_dataset = random_split(
        dataset, [train_size, validation_size]
    )

    ## Training loop
    n_epochs = 100
    batch_size = 16

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size)

    epoch_train_loss = [None] * len(train_loader)
    epoch_validation_loss = [None] * len(validation_loader)

    train_loss = [None] * n_epochs
    validation_loss = [None] * n_epochs
    
    # Model and optimizer
    feature_dim = dataset.n_features
    vae = VariationalAutoencoder(feature_dim, 3)
    vae.to(torch.double)

    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3) 

    for epoch in range(n_epochs):

        vae.train()
        for i, batch in enumerate(train_loader):
            
            # Forward pass
            output = vae(batch)
            px, pz, qz, z = [output[k] for k in ["px", "pz", "qz", "z"]]
            kl = kl_divergence(Independent(qz, 1),  Independent(pz, 1))
            loss = -px.log_prob(batch).sum(-1) + kl
            loss = loss.mean()
            optimizer.zero_grad()
            # loss, diagnostics, outputs = vi(vae, batch)
            ### Additional code for sampled kl?
            loss.backward()
            optimizer.step()

            epoch_train_loss[i] = loss.item()

        # Validating model
        vae.eval()
        with torch.no_grad():

            for i, batch in enumerate(validation_loader): 
                # Forward pass
                loss, diagnostics, outputs = vi(vae, batch)

                epoch_validation_loss[i] = loss.item()

        train_loss[epoch] = sum(epoch_train_loss)/len(epoch_train_loss)
        validation_loss[epoch] = sum(epoch_validation_loss)/len(epoch_validation_loss)

        print(f'train loss: {train_loss[epoch]:.4f}')
        print(f'validation loss: {validation_loss[epoch]:.4f}')      

    torch.save(vae.state_dict(), 'models/vae.pt')

        
        
# %%
