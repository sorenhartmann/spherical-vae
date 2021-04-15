from src.models.common import train_model
from torch.distributions.kl import kl_divergence
from torch.utils.data.dataset import random_split
from src.models.svae import SphericalVAE
from src.models.vae import VariationalAutoencoder

from torch.distributions import Independent

from src.data import MotionCaptureDataset

import torch

if __name__ == "__main__":

    torch.manual_seed(123)

    dataset = MotionCaptureDataset("07")

    svae = SphericalVAE(dataset.n_features, 3)
    vae = VariationalAutoencoder(dataset.n_features, 3)
    svae.to(torch.double)
    vae.to(torch.double)

    train_model(svae, dataset, n_epochs=1000, checkpoint_path="models/mocap/svae.pt")
    train_model(vae, dataset, n_epochs=1000, checkpoint_path="models/mocap/vae.pt")
