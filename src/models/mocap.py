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

    train_size = int(0.8 * len(dataset))
    validation_size = len(dataset) - train_size

    train_dataset, validation_dataset = random_split(
        dataset, [train_size, validation_size]
    )

    n_features = dataset.n_features

    svae = SphericalVAE(n_features, 3)
    vae = VariationalAutoencoder(n_features, 3)
    svae.to(torch.double)
    vae.to(torch.double)
    

    ## Training loop
    n_epochs = 1000
    batch_size = 8

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=batch_size
    )

    train_loss_svae = [None] * n_epochs
    validation_loss_svae = [None] * n_epochs
    train_loss_vae = [None] * n_epochs
    validation_loss_vae = [None] * n_epochs

    epoch_train_loss_svae = [None] * len(train_loader)
    epoch_validation_loss_svae = [None] * len(validation_loader)
    epoch_train_loss_vae = [None] * len(train_loader)
    epoch_validation_loss_vae = [None] * len(validation_loader)

    optimizer_svae = torch.optim.Adam(svae.parameters(), lr=1e-5)
    optimizer_vae = torch.optim.Adam(vae.parameters(), lr=1e-5)

    for epoch in range(n_epochs):

        svae.train()
        vae.train()

        for i, batch in enumerate(train_loader):
            # VAE
            output = vae(batch)
            px, pz, qz, z = [output[k] for k in ["px", "pz", "qz", "z"]]

            loss = -px.log_prob(batch).sum(-1) + kl_divergence(
                Independent(qz, 1), Independent(pz, 1)
                )
            loss = loss.mean()
            optimizer_vae.zero_grad()
            loss.backward()
            optimizer_vae.step()
            epoch_train_loss_vae[i] = loss.item()

            # SVAE
            output = svae(batch)
            px, pz, qz, z = [output[k] for k in ["px", "pz", "qz", "z"]]
            loss = -px.log_prob(batch).sum(-1) + kl_divergence(qz, pz)
            print(qz.k)
            if loss.isinf().any():
                raise ValueError
            loss = loss.mean()
            optimizer_svae.zero_grad()
            loss.backward()
            optimizer_svae.step()
            epoch_train_loss_svae[i] = loss.item()

        print(qz.k)

        # Validating model
        svae.eval()

        with torch.no_grad():

            for i, batch in enumerate(validation_loader):

                # SVAE
                output = svae(batch)
                px, pz, qz, z = [output[k] for k in ["px", "pz", "qz", "z"]]
                loss = -px.log_prob(batch).sum(-1) + kl_divergence(qz, pz)
                loss = loss.mean()
                epoch_validation_loss_svae[i] = loss.item()

                # VAE
                output = vae(batch)
                px, pz, qz, z = [output[k] for k in ["px", "pz", "qz", "z"]]
                loss = -px.log_prob(batch).sum(-1) +kl_divergence(
                Independent(qz, 1), Independent(pz, 1)
                )
                loss = loss.mean()
                epoch_validation_loss_vae[i] = loss.item()

        train_loss_svae[epoch] = sum(epoch_train_loss_svae) / len(epoch_train_loss_svae)
        validation_loss_svae[epoch] = sum(epoch_validation_loss_svae) / len(epoch_validation_loss_svae)
        train_loss_vae[epoch] = sum(epoch_train_loss_vae) / len(epoch_train_loss_vae)
        validation_loss_vae[epoch] = sum(epoch_validation_loss_vae) / len(epoch_validation_loss_vae)

        print(f"[SVAE] train loss: {train_loss_svae[epoch]:.4f}")
        print(f"[SVAE] validation loss: {validation_loss_svae[epoch]:.4f}")
        print(f"[VAE]  train loss: {train_loss_vae[epoch]:.4f}")
        print(f"[VAE ]validation loss: {validation_loss_vae[epoch]:.4f}")


    torch.save(svae.state_dict(), "models/mocap/svae.pt")
    torch.save(vae.state_dict(), "models/mocap/vae.pt")


    # %%
