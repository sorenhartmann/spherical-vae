from pathlib import Path
from src.models.common import BetaFunction, ModelTrainer
import click
import torch
from torch.utils.data.dataset import random_split
from src.models.svae import SphericalVAE
from src.data.mocap import MotionCaptureDataset
from torch.distributions.kl import kl_divergence

from src.data.synthetic import SyntheticS2

cuda = torch.cuda.is_available()
if cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class SphericalVAEWithoutCorrection(SphericalVAE):
    def get_loss(self, batch, return_kl=False, beta=1.0):

        output = self(batch)
        px, pz, qz, z = [output[k] for k in ["px", "pz", "qz", "z"]]
        kl_term = kl_divergence(qz, pz)

        loss = -px.log_prob(batch) + beta * kl_term

        if not return_kl:
            return loss.mean()
        else:
            return loss.mean(), kl_term.mean()


layer_sizes = [150]
latent_dim = 3
n_features = 62

model_args = {
    "latent_dim" : latent_dim,
    "feature_dim": n_features,
    "encoder_params": {
        "layer_sizes": layer_sizes,
        "dropout": 0.1,
        "activation_function": "Tanh",
    },
    "decoder_params": {
        "layer_sizes": layer_sizes[::-1],
        "dropout": 0.1,
        "activation_function": "Tanh",
    },
}

@click.command()
@click.option("--train-split", type=float, default=0.7, show_default=True)
@click.option("--batch-size", type=int, default=16, show_default=True)
@click.option("--n-models", type=int, default=5)
@click.option("--n-epochs", type=int, default=1000)
@click.option("--lr", type=float, default=5e-4)
@click.option("--beta_0", type=float, default= 1.)
@click.option("--run-name", type=str)
def main(
   train_split, batch_size, n_models, n_epochs, lr, beta_0, run_name
):

    torch.manual_seed(123)

    if run_name is None:
        run_name = "gradients"

    run_dir = Path(__file__).parents[2] / "runs" /  run_name

    dataset = MotionCaptureDataset("07")
    dataset.to(device)
    train_size = int(train_split * len(dataset))
    validation_size = len(dataset) - train_size
    train_dataset, validation_dataset = random_split(
        dataset,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(42),
    )

    trainer_args = {
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "beta_function": BetaFunction(beta_0, n_epochs),
    }

    train_args = {
        "train_dataset": train_dataset,
        "validation_dataset": validation_dataset,
        "random_state": None,
        "progress_bar": True,
    }
    


    for i in range(n_models):

        svae = SphericalVAE(**model_args)
        svae.to(device)

        mt = ModelTrainer(svae, **trainer_args, tb_dir=run_dir / f"w_corr_{i}")
        mt.train(**train_args)


        svae_wo_corr = SphericalVAEWithoutCorrection(**model_args)
        svae_wo_corr.to(device)
        
        mt = ModelTrainer(svae_wo_corr, **trainer_args, tb_dir=run_dir / f"wo_corr_{i}")
        mt.train(**train_args)
        


if __name__ == "__main__":

    main()