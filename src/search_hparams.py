
from src.models.common import ModelParameterError, train_model
from src.models.vae import VariationalAutoencoder
from src.models.svae import SphericalVAE

import click
from src.data import SyntheticS2, SkinCancerDataset, MotionCaptureDataset
import optuna
import torch

cuda = torch.cuda.is_available()
if cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

hparam_search_space = {
    "n_layers" : (1, 5),
    "layer_size" : (100, 1000),
    "lr" : (1e-4, 1e-2),
    "dropout" : (0.1, 0.5),
}

models = {
    "vae" : VariationalAutoencoder,
    "svae" : SphericalVAE,
}

datasets = {
    "synthetic" : SyntheticS2,
    "mocap" : MotionCaptureDataset,
}

def get_objective(model_type, data_type):

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

        #hov
        dataset = Dataset("07")
        # dataset = Dataset()
        dataset.to(device)

        dropout = trial.suggest_float("dropout", *dropout_range) 

        model = Model(
            feature_dim=dataset.n_features,
            latent_dim=3,
            encoder_params={"layer_sizes": encoder_layers_sizes, "dropout":dropout},
            decoder_params={"layer_sizes": decoder_layers_sizes, "dropout":dropout},
        )
        model.to(device)

        lr = trial.suggest_loguniform("lr", *lr_range)

        train_loss, validation_loss = train_model(
            model,
            dataset,
            label=f"{data_type}_{model_type}",
            n_epochs=1500,
            batch_size=8,
            lr=lr,
            trial=trial,
            progress_bar=False
        )

        return min(validation_loss)

    return objective


@click.command()
@click.argument("model", type=click.Choice(models.keys()))
@click.argument("data",type=click.Choice(datasets.keys()))
@click.option("--seed", type=int)
def main(model, data, seed):

    if seed is None:
        seed = 10

    study_name = f"{model}-{data}" 
    storage_name = f"sqlite:///runs/{study_name}.db"

    pruner = optuna.pruners.MedianPruner(n_warmup_steps=50, interval_steps=5)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction='minimize', 
        sampler=sampler, 
        pruner=pruner,
        load_if_exists=True,
        )
    objective = get_objective(model, data)
    study.optimize(objective, n_trials=50, catch=(ModelParameterError,))


if __name__ == "__main__":

    main()