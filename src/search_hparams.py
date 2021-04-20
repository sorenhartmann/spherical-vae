import multiprocessing
from multiprocessing import process
import os
from src.models.common import ModelParameterError, train_model
from src.models.vae import VariationalAutoencoder
from src.models.svae import SphericalVAE
import time
import click
from src.data import SyntheticS2, SkinCancerDataset, MotionCaptureDataset
import optuna
import torch
from multiprocessing import Pool

cuda = torch.cuda.is_available()
if cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

hparam_search_space = {
    "n_layers": (1, 5),
    "layer_size": (100, 1000),
    "lr": (1e-4, 1e-2),
    "dropout": (0.1, 0.5),
}

models = {
    "vae": VariationalAutoencoder,
    "svae": SphericalVAE,
}

datasets = {
    "synthetic": SyntheticS2,
    "mocap": MotionCaptureDataset,
}


def get_objective(model_type, data_type, n_epochs):

    Model = models[model_type]
    Dataset = datasets[data_type]

    def objective(trial: optuna.trial.Trial):

        n_layers_range = hparam_search_space["n_layers"]
        layer_size_range = hparam_search_space["layer_size"]
        lr_range = hparam_search_space["lr"]
        dropout_range = hparam_search_space["dropout"]

        encoder_n_layers = trial.suggest_int("n_layers", *n_layers_range)
        encoder_layers_sizes = []
        for i in range(encoder_n_layers):
            encoder_layers_sizes.append(
                trial.suggest_int(f"layer_size_{i+1}", *layer_size_range)
            )

        decoder_layers_sizes = encoder_layers_sizes[::-1]

        # hov
        dataset = Dataset("07")
        # dataset = Dataset()
        dataset.to(device)

        dropout = trial.suggest_float("dropout", *dropout_range)

        model = Model(
            feature_dim=dataset.n_features,
            latent_dim=3,
            encoder_params={"layer_sizes": encoder_layers_sizes, "dropout": dropout},
            decoder_params={"layer_sizes": decoder_layers_sizes, "dropout": dropout},
        )
        model.to(device)

        lr = trial.suggest_loguniform("lr", *lr_range)

        train_loss, validation_loss = train_model(
            model,
            dataset,
            label=f"{data_type}_{model_type}",
            n_epochs=n_epochs,
            batch_size=16,
            lr=lr,
            trial=trial,
            progress_bar=False,
        )

        loss = min(validation_loss)

        return loss

    return objective

def _load_and_run(study_name, storage_name, objective, n_trials, seed):

    pruner = optuna.pruners.MedianPruner(n_warmup_steps=50, interval_steps=5)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.load_study(
        study_name=study_name,
        storage=storage_name,
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        catch=(ModelParameterError,),
    )


@click.command()
@click.argument("model", type=click.Choice(models.keys()))
@click.argument("data", type=click.Choice(datasets.keys()))
@click.option("--n-trials", type=int)
@click.option("--seed", type=int)
@click.option("--processes", type=int)
@click.option("--epochs", type=int)
def main(model, data, n_trials, seed, processes, epochs):

    if n_trials is None:
        n_trials = 100

    if seed is None:
        seed = 10

    if epochs is None:
        epochs = 100

    study_name = f"{data}-{model}"
    storage_name = f"sqlite:///runs/{study_name}.db"

    optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="minimize",
        load_if_exists=True,
    )

    objective = get_objective(model, data, epochs)

    if processes == 1 or processes is None:
        _load_and_run(study_name, storage_name, objective, n_trials, seed)
    else:
        n_processes = os.cpu_count() if processes == -1 else processes
        for i in range(n_processes):
            print(f"Starting process {i}")
            p = multiprocessing.Process(
                target=_load_and_run,
                args=(
                    study_name,
                    storage_name,
                    objective,
                    n_trials,
                    seed + i,
                ),
            )
            time.sleep(1)
            p.start()


if __name__ == "__main__":

    main()