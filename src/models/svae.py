import torch
import numpy as np
from torch.distributions import Distribution, Normal
from torch.nn import Module, Linear, ReLU, Sequential
from torch.tensor import Tensor
from src.distributions import VonMisesFisher, SphereUniform
from sklearn.model_selection import train_test_split

from src.data.genSynthData import genNoisySynthDataS2
from plotly import graph_objects as go

import random
class SphericalVAE(Module):

    def __init__(self, feature_dim, latent_dim):
        
        super().__init__()

        self.feature_dim = feature_dim
        self.latent_dim = latent_dim

        self.encoder = Sequential(
            Linear(in_features=feature_dim, out_features=256),
            ReLU(),
            Linear(in_features=256, out_features=128),
            ReLU(),
            Linear(in_features=128, out_features=64), 
            ReLU(),
            Linear(in_features=64, out_features=32), 
            ReLU(),
            Linear(in_features=32, out_features=latent_dim + 1)
        )

        self.decoder = Sequential(
            Linear(in_features=latent_dim, out_features=128),
            ReLU(),
            Linear(in_features=128, out_features=256),
            ReLU(),
            Linear(in_features=256, out_features=512), 
            ReLU(),
            Linear(in_features=512, out_features=2*self.feature_dim)
        )

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
    
        return {'px': px, 'pz': pz, 'qz': qz, 'z': z}

# %% 
if __name__ == "__main__":

    # Splitting synthetic data into train and validation 
    X, y = genNoisySynthDataS2(plotCartesian=False, Nsamples=100)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3, random_state = 62) 

    X_train = torch.tensor(X_train, dtype=torch.float)
    X_val = torch.tensor(X_val, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.float)
    y_val = torch.tensor(y_val, dtype=torch.float)
    
    ## Training loop
    n_epochs = 100
    batch_size = 16

    n_batches_train = int(np.ceil(X_train.shape[0]/batch_size))
    n_batches_validation = int(np.ceil(X_val.shape[0]/batch_size))

    epoch_train_loss = [None]*n_batches_train
    epoch_validation_loss = [None]*n_batches_validation
    train_loss = [None]*n_epochs
    validation_loss = [None]*n_epochs
    
    # Model and optimizer
    feature_dim = X.shape[-1]
    svae = SphericalVAE(feature_dim=feature_dim, latent_dim=3)

    optimizer = torch.optim.Adam(svae.parameters(), lr=1e-3) 
    for epoch in range(n_epochs):

        trainloader = torch.utils.data.DataLoader(X_train, batch_size=batch_size)
        trainloader = iter(trainloader)
        validationloader = torch.utils.data.DataLoader(X_val, batch_size=batch_size)
        validationloader = iter(validationloader)

        svae.train()

        for i, batch in enumerate(trainloader):
            
            # Forward pass
            output = svae.forward(batch)
            loss = -output["px"].log_prob(batch).sum() + sum(output['qz'].k) / 100 #Poor mans kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss[i] = loss.item()

        # Validating model
        svae.eval()
        with torch.no_grad():

            for i, batch in enumerate(validationloader): 
                # Forward pass
                output = svae.forward(batch)
                loss = -output["px"].log_prob(batch).sum() +  sum(output['qz'].k) / 100

                epoch_validation_loss[i] = loss.item()

        train_loss[epoch] = sum(epoch_train_loss)/len(epoch_train_loss)
        validation_loss[epoch] = sum(epoch_validation_loss)/len(epoch_validation_loss)
    
        print(f'train loss: {train_loss[epoch]:.2f}')
        print(f'validation loss: {validation_loss[epoch]:.2f}')        
        
        
# %%
