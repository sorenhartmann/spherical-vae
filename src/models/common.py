from logging import log
import os
from pathlib import Path
import torch
from torch.nn import Module, Linear, Sequential, Dropout


from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
from tqdm import trange
from datetime import datetime
import optuna

class ModelParameterError(Exception):
    pass

class Encoder(Module):
    def __init__(
        self, in_features, out_features, layer_sizes=None, activation_function=None, dropout=None
    ):

        super().__init__()

        self.dropout = dropout

        if layer_sizes is None:
            layer_sizes = [100, 100]
        self._layer_sizes = layer_sizes 

        if activation_function is None:
            activation_function = "ReLU"
        self._activation_function = activation_function

        ActFunc = getattr(torch.nn.modules.activation, activation_function)

        layers = []
        in_size = in_features
        for out_size in layer_sizes:
            layers.append(Linear(in_size, out_size))
            layers.append(ActFunc())
            if dropout is not None:
                layers.append(Dropout(dropout))
            in_size = out_size
        layers.append(Linear(in_size, out_features))

        self.ffnn = Sequential(*layers)

    def forward(self, x):

        return self.ffnn(x)


class Decoder(Encoder):
    pass

class SummaryWriter(SummaryWriter):
    def add_hparams(self, hparam_dict, metric_dict):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        logdir = self._get_file_writer().get_logdir()
        
        with SummaryWriter(log_dir=logdir) as w_hp:
            w_hp.file_writer.add_summary(exp)
            w_hp.file_writer.add_summary(ssi)
            w_hp.file_writer.add_summary(sei)
            for k, v in metric_dict.items():
                w_hp.add_scalar(k, v)


def train_model(
    model,
    dataset,
    label=None,
    train_size=0.8,
    n_epochs=100,
    batch_size=16,
    lr=1e-3,
    checkpoint_path=None,
    retrain=False,
    epoch_callback=None,
    trial:optuna.trial.Trial=None,
    progress_bar=True,
):

    ## Can only train models with defined loss
    assert hasattr(model, "get_loss") and callable(
        model.get_loss
    ), "Model needs to have implemented a .get_loss method"

    ## For tensorboard integration
    if label is not None:
        now = datetime.now().strftime("%b%d_%H-%M-%S")
        log_dir = os.path.join("runs", label, now)
        if trial is not None:
            log_dir = f"{log_dir}_{trial.number}"
        writer = SummaryWriter(flush_secs=5, log_dir=log_dir)
    else:
        writer=None

    def add_scalar(*args, **kwargs):
        if writer is not None:
            writer.add_scalar(*args, **kwargs)
        return

    ## Don't retrain model if already trained TODO: Check hyperparameters?
    if checkpoint_path is not None and Path(checkpoint_path).exists() and not retrain:
        return

    ## Set seed for train/test split TODO: Maybe do outside function?
    torch.manual_seed(123)

    train_size = int(train_size * len(dataset))
    validation_size = len(dataset) - train_size

    train_dataset, validation_dataset = random_split(
        dataset, [train_size, validation_size]
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=batch_size
    )

    epoch_train_loss = [None] * len(train_loader)
    epoch_validation_loss = [None] * len(validation_loader)

    train_loss = [None] * n_epochs
    validation_loss = [None] * n_epochs

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    iter_ = trange(n_epochs) if progress_bar else range(n_epochs)

    for epoch in iter_:

        # eg. for annealing beta parameter FIXME: Currently a bit wonky with the 
        # required arguments. Would probably be more intutive with the trainer wrapped 
        # in an object -> could just be fixed with subclassing
        if epoch_callback is not None:
            epoch_callback(model, epoch, n_epochs, writer=writer)

        model.train()
        for i, batch in enumerate(train_loader):

            # Forward pass
            loss = model.get_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss[i] = loss.item()

        # Validating model
        model.eval()
        with torch.no_grad():

            for i, batch in enumerate(validation_loader):
                # Forward pass
                loss = model.get_loss(batch)
                epoch_validation_loss[i] = loss.item()

        train_loss[epoch] = sum(epoch_train_loss) / len(epoch_train_loss)
        add_scalar("Loss/Train", train_loss[epoch], epoch)
        validation_loss[epoch] = sum(epoch_validation_loss) / len(epoch_validation_loss)
        add_scalar("Loss/Validation", validation_loss[epoch], epoch)

        if trial is not None:
            trial.report(validation_loss[epoch], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned

    if writer is not None:
        hparams = {"lr" : lr, "batch_size": batch_size}
        hparams.update(get_hparams(model))
        metrics = {"neg_ELBO" : min(validation_loss) }
        writer.add_hparams(hparams, metrics)
        writer.close()

    if checkpoint_path is not None:
        torch.save(model.state_dict(), checkpoint_path)

    return train_loss, validation_loss

def get_hparams(model):
    return {
        "latent_dim" : model.latent_dim,
        "encoder__layer_sizes": str(model.encoder._layer_sizes),
        "encoder__n_layers": len(model.encoder._layer_sizes),
        "encoder__activation_function": model.encoder._activation_function,
        "encoder__dropout": model.encoder.dropout,
        "decoder__layer_sizes": str(model.decoder._layer_sizes),
        "decoder__n_layers": len(model.decoder._layer_sizes),
        "decoder__activation_function": model.decoder._activation_function,
        "decoder__dropout": model.decoder.dropout,
    }
