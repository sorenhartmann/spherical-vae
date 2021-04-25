import multiprocessing
import os
from pathlib import Path

from torch.utils.data.dataset import Subset, random_split
from src.models.common import ModelParameterError, ModelTrainer
from src.models.vae import VariationalAutoencoder
from src.models.svae import SphericalVAE
import time
import click
from src.data import SyntheticS2, MotionCaptureDataset
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
    "beta_0": (1e-3, 1),
}


def get_model_class(model_name):

    if model_name == "vae":
        return VariationalAutoencoder
    elif model_name == "svae":
        return SphericalVAE
    else:
        raise ValueError


def get_dataset(dataset_name):

    if dataset_name == "synthetic":
        return SyntheticS2()
    elif "mocap" in dataset_name:
        _, subject_id = dataset_name.split("-")
        return MotionCaptureDataset(subject_id)
    else:
        raise ValueError


class BetaFunction:
    def __init__(self, beta_0, n_epochs, start=0.25, end=0.75) -> None:

        self.beta_0 = beta_0
        self.n_epochs = n_epochs
        self.start = start * n_epochs
        self.end = end * n_epochs

        self.a = 2 * (1 - beta_0) / self.n_epochs
        self.b = beta_0 - self.a * self.n_epochs / 4

        self.elbo_valid_after = self.end

    def __call__(self, i):

        if i < self.start:
            return self.beta_0
        elif i >= self.start and i < self.end:
            return self.a * i + self.b
        else:
            return 1


class Objective:

    def __init__(
        self,
        model_name,
        train_dataset,
        validation_dataset,
        n_epochs=100,
        batch_size=16,
        latent_dim=3,
        log_dir=None,
        checkpoint_dir=None,
        keep_best=10,
    ):

        self.n_layers_range = hparam_search_space["n_layers"]
        self.layer_size_range = hparam_search_space["layer_size"]
        self.lr_range = hparam_search_space["lr"]
        self.dropout_range = hparam_search_space["dropout"]
        self.beta_0_range = hparam_search_space["beta_0"]

        self.ModelClass = get_model_class(model_name)
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.latent_dim = latent_dim

        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.keep_best = keep_best

    def __call__(self, trial: optuna.trial.Trial):

        encoder_n_layers = trial.suggest_int("n_layers", *self.n_layers_range)
        encoder_layers_sizes = []
        for i in range(encoder_n_layers):
            encoder_layers_sizes.append(
                trial.suggest_int(f"layer_size_{i+1}", *self.layer_size_range)
            )

        decoder_layers_sizes = encoder_layers_sizes[::-1]

        dropout = trial.suggest_float("dropout", *self.dropout_range)

        if type(self.train_dataset) is Subset:
            n_features = self.train_dataset.dataset.n_features
        else:
            n_features = self.train_dataset.n_features

        model = self.ModelClass(
            feature_dim=n_features,
            latent_dim=self.latent_dim,
            encoder_params={
                "layer_sizes": encoder_layers_sizes,
                "dropout": dropout,
                "activation_function": "Tanh",
            },
            decoder_params={
                "layer_sizes": decoder_layers_sizes,
                "dropout": dropout,
                "activation_function": "Tanh",
            },
        )
        model.to(device)

        for dataset in [self.train_dataset, self.validation_dataset]:
            if type(dataset) is Subset:
                dataset.dataset.to(device)
            else:
                dataset.to(device)

        lr = trial.suggest_loguniform("lr", *self.lr_range)

        beta_0 = trial.suggest_loguniform("beta_0", *self.beta_0_range)
        beta_function = BetaFunction(beta_0, self.n_epochs)

        if self.log_dir is not None:
            tb_dir = self.log_dir / f"{trial.number:03}"
        else:
            tb_dir = None

        if self.checkpoint_dir is not None:
            checkpoint_path = self.checkpoint_dir / f"{trial.number:03}.pt"
        else:
            checkpoint_path = None

        model_trainer = ModelTrainer(
            model=model,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            lr=lr,
            beta_function=beta_function,
            tb_dir=tb_dir,
            checkpoint_path=checkpoint_path
        )

        model_trainer.train(
            train_dataset=self.train_dataset,
            validation_dataset=self.validation_dataset,
            random_state=None,
            progress_bar=False,
        )

        loss = min(model_trainer.validation_loss)

        trial.model_ = model

        return loss

    def callback(self, study, trial):

        sorted_trials = sorted((trial for trial in study.trials if trial.value is not None), key= lambda x: x.value)
        trials_to_keep = set(trial.number for trial in sorted_trials[:self.keep_best])
        for checkpoint in Path(self.checkpoint_dir).iterdir():
            if int(checkpoint.stem) not in trials_to_keep:
                checkpoint.unlink()



def _load_and_run(study_name, storage_name, objective, n_trials, seed):

    pruner = optuna.pruners.MedianPruner(n_warmup_steps=n_trials//3, interval_steps=5)
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
        callbacks=[objective.callback],
    )


@click.command()
@click.argument("model", default="svae")
@click.argument("data", default="synthetic")
@click.option("--n-trials", type=int, default=100, show_default=True)
@click.option("--n-processes", type=int, default=1, show_default=True)
@click.option("--n-epochs", type=int, default=100, show_default=True)
@click.option("--train-split", type=float, default=0.7, show_default=True)
@click.option("--batch-size", type=int, default=16, show_default=True)
@click.option("--latent_dim", type=int, default=3, show_default=True)
@click.option("--seed", type=int, default=10)
@click.option("--keep-best", type=int, default=10)
def main(
    model,
    data,
    n_trials,
    n_processes,
    n_epochs,
    train_split,
    batch_size,
    latent_dim,
    seed,
    keep_best,
):

    torch.manual_seed(seed)

    dataset = get_dataset(dataset_name=data)
    train_size = int(train_split * len(dataset))
    validation_size = len(dataset) - train_size
    train_dataset, validation_dataset = random_split(
        dataset,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(42),
    )

    run_dir = (Path(__file__).parents[1] / "runs").resolve()

    study_name = f"{data}-{model}"
    study_dir = run_dir / study_name

    if not study_dir.exists():
        study_dir.mkdir(parents=True)
    storage_name = (
        f"sqlite:///{(study_dir / 'optuna-storage.db').relative_to(os.getcwd())}"
    )

    optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="minimize",
        load_if_exists=True,

    )

    log_dir = study_dir / "hp-search"
    checkpoint_dir = study_dir / "checkpoints"

    objective = Objective(
        model_name=model,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        n_epochs=n_epochs,
        batch_size=batch_size,
        latent_dim=latent_dim,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        keep_best=keep_best
    )

    if n_trials == -1:
        n_trials = None

    if n_processes == 1 or n_processes is None:
        _load_and_run(study_name, storage_name, objective, n_trials, seed)
    else:
        n_processes = os.cpu_count() if n_processes == -1 else n_processes
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