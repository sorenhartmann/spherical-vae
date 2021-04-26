from src.models.common import ModelTrainer
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

    train_size = int(0.7 * len(dataset))
    validation_size = len(dataset) - train_size
    train_dataset, validation_dataset = random_split(
        dataset,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(42),
    )

    n_features = dataset.n_features
    latent_dim = 3
    encoder_layers_sizes = [100, 100, 100]
    decoder_layers_sizes = [100, 100, 100]
    dropout = 0.2

    model_params = {
        "feature_dim": n_features,
        "latent_dim": latent_dim,
        "encoder_params": {
            "layer_sizes": encoder_layers_sizes,
            "dropout": dropout,
            "activation_function": "Tanh",
        },
        "decoder_params": {
            "layer_sizes": decoder_layers_sizes,
            "dropout": dropout,
            "activation_function": "Tanh",
        },
    }
    svae = SphericalVAE(**model_params)
    vae = VariationalAutoencoder(**model_params)

    mt_svae = ModelTrainer(svae, n_epochs=1000, tb_label="mocap-svae", checkpoint_path="models/mocap/svae.pt")
    mt_svae.train(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
    )
    mt_vae = ModelTrainer(vae, n_epochs=1000, tb_label="mocap-vae", checkpoint_path="models/mocap/vae.pt")
    mt_vae.train(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
    )