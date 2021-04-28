import multiprocessing
import os
from pathlib import Path
from src.data.skin_cancer import SkinCancerDataset

from torch.utils.data.dataset import Subset, random_split
from src.models.common import BetaFunction, ModelParameterError, ModelTrainer
from src.models.vae import VariationalAutoencoder
from src.models.svae import SphericalVAE
from src.models.convolutional import ConvVariationalAutoencoder, ConvSphericalVAE
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

def get_model_class(model_name, is_conv=False):

    if model_name == "vae" and not is_conv:
        return VariationalAutoencoder
    elif model_name == "vae" and is_conv:
        return ConvVariationalAutoencoder
    elif model_name == "svae" and not is_conv:
        return SphericalVAE
    elif model_name == "svae" and is_conv:
        return ConvSphericalVAE


def get_dataset(dataset_name):

    if dataset_name == "synthetic":
        return SyntheticS2()
    elif "mocap" in dataset_name:
        _, subject_id = dataset_name.split("-")
        return MotionCaptureDataset(subject_id)
    elif dataset_name == "skin-cancer":
        return SkinCancerDataset(image_size=(225, 300))


class Objective:
    def __init__(
        self,
        model_name,
        train_dataset,
        validation_dataset,
        n_epochs=100,
        log_dir=None,
        checkpoint_dir=None,
        keep_best=10,
        batch_size=16,
        latent_dim=3,
    ):

        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.latent_dim = latent_dim

        if type(train_dataset) is Subset:
            if hasattr(self.train_dataset.dataset, "n_features"):
                self.n_features = self.train_dataset.dataset.n_features
            if hasattr(self.train_dataset.dataset, "image_size"):
                self.image_size = self.train_dataset.dataset.image_size
            self.dataset_name = self.train_dataset.dataset.name
        else:
            if hasattr(self.train_dataset, "n_features"):
                self.n_features = self.train_dataset.n_features
            if hasattr(self.train_dataset, "image_size"):
                self.image_size = self.train_dataset.image_size
            self.dataset_name = self.train_dataset.name

        self.ModelClass = get_model_class(
            model_name, is_conv=self.dataset_name == "skin-cancer"
        )

        for dataset in [self.train_dataset, self.validation_dataset]:
            if type(dataset) is Subset:
                dataset.dataset.to(device)
            else:
                dataset.to(device)

        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.keep_best = keep_best

    def regular_objective(self, trial):

        # ----- Suggest parameters -----
        n_layers = trial.suggest_int("n_layers", 1, 5)
        lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        beta_0 = trial.suggest_loguniform("beta_0", 1e-3, 1)
        layer_sizes = []
        for i in range(n_layers):
            layer_sizes.append(trial.suggest_int(f"layer_size_{i+1}", 100, 1000))

        # ----- Setup model ----
        model = self.ModelClass(
            feature_dim=self.n_features,
            latent_dim=self.latent_dim,
            encoder_params={
                "layer_sizes": layer_sizes,
                "dropout": dropout,
                "activation_function": "Tanh",
            },
            decoder_params={
                "layer_sizes": layer_sizes[::-1],
                "dropout": dropout,
                "activation_function": "Tanh",
            },
        )
        model.to(device)

        # ----- Setup Tensorboard and checkpoints -----
        if self.log_dir is not None:
            tb_dir = self.log_dir / f"{trial.number:03}"
        else:
            tb_dir = None
        if self.checkpoint_dir is not None:
            checkpoint_path = self.checkpoint_dir / f"{trial.number:03}.pt"
        else:
            checkpoint_path = None

        # ----- Train model ----
        model_trainer = ModelTrainer(
            model=model,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            lr=lr,
            beta_function=BetaFunction(beta_0, self.n_epochs),
            tb_dir=tb_dir,
            checkpoint_path=checkpoint_path,
            trial=trial,
        )
        model_trainer.train(
            train_dataset=self.train_dataset,
            validation_dataset=self.validation_dataset,
            random_state=None,
            progress_bar=False,
        )

        # ----- Report loss -----
        loss = min(model_trainer.validation_loss)
        return loss

    def conv_objective(self, trial):

        # ----- Suggest parameters -----
        ffnn_layers = trial.suggest_int("n_ffnn_layers", 1, 5)
        lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        beta_0 = trial.suggest_loguniform("beta_0", 1e-3, 1)
        ffnn_layers_size = []
        for i in range(ffnn_layers):
            ffnn_layers_size.append(
                trial.suggest_int(f"ffnn_layers_size_{i+1}", 100, 1000)
            )

        conv_layers = trial.suggest_int("n_conv_layers", 1, 4)

        encoder_kernel_size = []
        encoder_stride = []
        encoder_out_channel_size = []
        encoder_padding_size = []

        for i in range(conv_layers):
            encoder_kernel_size.append(trial.suggest_int(f"kernel_size_{i+1}", 2, 7))
            encoder_stride.append(trial.suggest_int(f"stride_{i+1}", 1, 5))
            encoder_out_channel_size.append(
                trial.suggest_int(f"out_channels_{i+1}", 2, 10)
            )
            encoder_padding_size.append(trial.suggest_int(f"padding_size_{i+1}", 0, 7))

        maxpool_kernel = trial.suggest_int("maxpool_kernel", 2, 4)
        maxpool_stride = trial.suggest_int("maxpool_stride", 2, 4)

        # ----- Setup model ----
        model = self.ModelClass(
            image_size=(3,) + self.image_size,
            latent_dim=self.latent_dim,
            encoder_params={
                "kernel_size": encoder_kernel_size,
                "stride": encoder_stride,
                "out_channel_size": encoder_out_channel_size,
                "padding_size": encoder_padding_size,
                "activation_function": "Tanh",
                "ffnn_layer_size": ffnn_layers_size,
                "dropout": dropout,
                "dropout2d": dropout,
                "maxpool_kernel": maxpool_kernel,
                "maxpool_stride": maxpool_stride,
            },
            decoder_params={
                "kernel_size": encoder_kernel_size[::-1],
                "stride": encoder_stride[::-1],
                "in_channel_size": encoder_out_channel_size[::-1],
                "ffnn_layer_size": ffnn_layers_size[::-1],
                "dropout": dropout,
                "dropout2d": dropout,
                "activation_function": "Tanh",
            },
        )
        model.to(device)

        # ----- Setup Tensorboard and checkpoints -----
        if self.log_dir is not None:
            tb_dir = self.log_dir / f"{trial.number:03}"
        else:
            tb_dir = None
        if self.checkpoint_dir is not None:
            checkpoint_path = self.checkpoint_dir / f"{trial.number:03}.pt"
        else:
            checkpoint_path = None

        # ----- Train model ----
        model_trainer = ModelTrainer(
            model=model,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            lr=lr,
            beta_function=BetaFunction(beta_0, self.n_epochs),
            tb_dir=tb_dir,
            checkpoint_path=checkpoint_path,
            trial=trial,
        )
        model_trainer.train(
            train_dataset=self.train_dataset,
            validation_dataset=self.validation_dataset,
            random_state=None,
            progress_bar=False,
        )

        # ----- Report loss -----
        loss = min(model_trainer.validation_loss)
        return loss

    def __call__(self, trial: optuna.trial.Trial):

        if self.dataset_name in ["mocap", "synthetic"]:
            return self.regular_objective(trial)
        elif self.dataset_name == "skin-cancer":
            return self.conv_objective(trial)

    def callback(self, study, trial):

        sorted_trials = sorted(
            (trial for trial in study.trials if trial.value is not None),
            key=lambda x: x.value,
        )
        trials_to_keep = set(trial.number for trial in sorted_trials[: self.keep_best])

        try:
            for checkpoint in Path(self.checkpoint_dir).iterdir():
                if int(checkpoint.stem) not in trials_to_keep:
                    checkpoint.unlink()
        except FileNotFoundError:
            pass


def _load_and_run(study_name, storage_name, objective, n_trials, seed):

    pruner = optuna.pruners.MedianPruner(n_warmup_steps=n_trials // 3, interval_steps=5)
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
@click.option("--latent-dim", type=int, default=3, show_default=True)
@click.option("--seed", type=int, default=10)
@click.option("--keep-best", type=int, default=10)
@click.option("--name", type=str)
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
    name,
):

    torch.manual_seed(seed)

    dataset = get_dataset(dataset_name=data)
    dataset.to(device)

    train_size = int(train_split * len(dataset))
    validation_size = len(dataset) - train_size
    train_dataset, validation_dataset = random_split(
        dataset,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(42),
    )

    run_dir = (Path(__file__).parents[1] / "runs").resolve()

    if name is None:
        study_name = f"{data}-{model}"
    else:
        study_name = name

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

    model_parameters = {}
    if batch_size is not None:
        model_parameters["batch_size"] = batch_size
    if latent_dim is not None:
        model_parameters["latent_dim"] = latent_dim

    objective = Objective(
        model_name=model,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        n_epochs=n_epochs,
        batch_size=batch_size,
        latent_dim=latent_dim,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        keep_best=keep_best,
    )

    if n_trials == -1:
        n_trials = None

    if n_processes == 1 or n_processes is None or device == torch.device("cuda"):
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