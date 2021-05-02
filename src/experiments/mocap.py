from pathlib import Path
from src.data.mocap import MotionCaptureDataset, split_time_series
from src.models.svae import SphericalVAE
from src.models.vae import VariationalAutoencoder
from src.models.common import ModelTrainer, BetaFunction
from torch.utils.data import ConcatDataset
import torch
import click

cuda = torch.cuda.is_available()
if cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

layer_sizes = [700, 750, 700]
latent_dim = 2
n_features = 62

model_args = {
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

def get_data(experiment_name, train_split):

    if experiment_name == "run-walk":
        
        datasets = [
            MotionCaptureDataset("09"),
            MotionCaptureDataset("08"),
        ]

    elif experiment_name == "walk-walk":

        datasets = [
            MotionCaptureDataset("07"),
            MotionCaptureDataset("08"),
        ]

    elif experiment_name == "dancing":

        datasets = [
            MotionCaptureDataset("60"),
            MotionCaptureDataset("94"),
        ]

    elif experiment_name == "swimming":
        
        datasets = [
            MotionCaptureDataset("126")
        ]

    else:
        raise ValueError

    train_datasets = []
    validation_datasets = []

    for dataset in datasets:
        dataset.to(device)
        train_dataset, validation_dataset = split_time_series(dataset, train_split)
        train_datasets.append(train_dataset)
        validation_datasets.append(validation_dataset)

    train_dataset_concat = ConcatDataset(train_datasets)
    validation_dataset_concat = ConcatDataset(validation_datasets)

    return train_dataset_concat, validation_dataset_concat


@click.command()
@click.argument("model-name", default="svae")
@click.argument("experiment-name", default="run-walk")
@click.option("--train-split", type=float, default=0.7, show_default=True)
@click.option("--batch-size", type=int, default=16, show_default=True)
@click.option("--n-models", type=int, default=5)
@click.option("--n-epochs", type=int, default=1000)
@click.option("--lr", type=float, default=1e-4)
@click.option("--beta_0", type=float, default=0.2)
def main(
    model_name, experiment_name, train_split, batch_size, n_models, n_epochs, lr, beta_0
):

    run_dir = Path(__file__).parents[2] / "runs" / f"{experiment_name}"
    run_dir.mkdir(exist_ok=True, parents=True)
    if (run_dir / "best_{model_name}.pt").exists():
        raise FileExistsError

    torch.manual_seed(123)

    if model_name == "vae":
        Model = VariationalAutoencoder
        latent_dim_ = latent_dim
    elif model_name == "svae":
        Model = SphericalVAE
        latent_dim_ = latent_dim + 1

    train_dataset, validation_dataset = get_data(experiment_name, train_split)

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

    best_loss = None

    for i in range(n_models):

        ## Fit some models, choose best
        model = Model(latent_dim=latent_dim_, **model_args)
        model.to(device)
        mt = ModelTrainer(model, **trainer_args, tb_dir=run_dir / f"{model_name}_{i}")
        mt.train(**train_args)
        loss = min(mt.validation_loss)
        if best_loss is None or loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), run_dir / "best_{model_name}.pt")


if __name__ == "__main__":
    main()