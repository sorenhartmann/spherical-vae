from logging import log
import os
from pathlib import Path
import torch
from torch.distributions.kl import kl_divergence
from torch.nn import Module, Linear, Sequential, Dropout
from scipy.special import ive

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
        self,
        in_features,
        out_features,
        layer_sizes=None,
        activation_function=None,
        dropout=None,
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
            raise TypeError("hparam_dict and metric_dict should be dictionary.")
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        logdir = self._get_file_writer().get_logdir()

        with SummaryWriter(log_dir=logdir) as w_hp:
            w_hp.file_writer.add_summary(exp)
            w_hp.file_writer.add_summary(ssi)
            w_hp.file_writer.add_summary(sei)
            for k, v in metric_dict.items():
                w_hp.add_scalar(k, v)

class ModelTrainer:
    def __init__(
        self,
        model,
        n_epochs=100,
        batch_size=16,
        lr=1e-3,
        beta_function=None,
        checkpoint_path=None,
        tb_label=None,
        tb_dir=None,
    ):

        self.model = model

        ## Can only train models with defined loss
        assert hasattr(self.model, "get_loss") and callable(
            self.model.get_loss
        ), "Model needs to have implemented a .get_loss method"

        ## Training parameters
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.beta_function = beta_function

        ## Path to saved trained model to
        self.checkpoint_path = checkpoint_path
        
        self.init_tb_writer(tb_dir=tb_dir, tb_label=tb_label )


    def init_tb_writer(self, tb_dir=None, tb_label=None):

        ## Label for tensorboard integration
        if tb_dir is None and tb_label is None:
            self.tb_writer = None
            return
        elif tb_dir is None and tb_label is not None:
            now = datetime.now().strftime("%b%d_%H-%M-%S")
            log_dir = os.path.join("runs", tb_label, now)
        elif tb_dir is not None:
            log_dir = tb_dir

        self.tb_writer = SummaryWriter(flush_secs=5, log_dir=log_dir)

    def train_setup(
        self,
        train_dataset,
        validation_dataset,
        random_state=123,
    ):

        if random_state is not None:
            torch.manual_seed(random_state)

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size
        )
        self.validation_loader = torch.utils.data.DataLoader(
            validation_dataset, batch_size=self.batch_size
        )

        self.train_loss = [None] * self.n_epochs
        self.validation_loss = [None] * self.n_epochs
        self.kl_divergence = [None] * self.n_epochs

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train(
        self,
        train_dataset,
        validation_dataset,
        random_state=123,
        progress_bar=True,
    ):
    
        self.train_setup(
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            random_state=random_state,
        )

        iter_ = trange(self.n_epochs) if progress_bar else range(self.n_epochs)

        for epoch in iter_:
            self.per_epoch(epoch)

        self.after_training()

    def train_loop(self, epoch):
        
        beta = 1. if self.beta_function is None else self.beta_function(epoch)

        self.model.train()
        for i, batch in enumerate(self.train_loader):

            # Forward pass
            loss = self.model.get_loss(batch, beta=beta)
            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()
            self.epoch_train_loss[i] = loss.item()

    def validation_loop(self, epoch):
        
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.validation_loader):
                # Forward pass
                loss, kl_term = self.model.get_loss(batch, return_kl=True)
                self.epoch_validation_loss[i] = loss.item()
                self.epoch_kl_divergence[i] = kl_term.item()

                    
    def per_epoch(self, epoch):

        self.epoch_train_loss = [None] * len(self.train_loader)
        self.epoch_validation_loss = [None] * len(self.validation_loader)
        self.epoch_kl_divergence = [None] * len(self.validation_loader)

        self.train_loop(epoch)
        self.validation_loop(epoch)

        self.train_loss[epoch] = sum(self.epoch_train_loss) / len(self.epoch_train_loss)
        self.validation_loss[epoch] = sum(self.epoch_validation_loss) / len(self.epoch_validation_loss)
        self.kl_divergence[epoch] = sum(self.epoch_kl_divergence) / len(self.epoch_kl_divergence)
        
        if self.tb_writer is not None:
            self.tb_writer.add_scalar("Loss/Train", self.train_loss[epoch], epoch)
            self.tb_writer.add_scalar("Loss/Validation", self.validation_loss[epoch], epoch)
            self.tb_writer.add_scalar("Average KL-term", self.kl_divergence[epoch], epoch)

    def after_training(self):

        if self.tb_writer is not None:
            hparams = {"lr": self.lr, "batch_size": self.batch_size, "beta_0": self.beta_function.beta_0}
            hparams.update(get_hparams(self.model))
            metrics = {"neg_ELBO": min(self.validation_loss)}
            self.tb_writer.add_hparams(hparams, metrics)
            self.tb_writer.close()

        if self.checkpoint_path is not None:
            torch.save(self.model.state_dict(), self.checkpoint_path)


def get_hparams(model):
    return {
        "latent_dim": model.latent_dim,
        "encoder__layer_sizes": str(model.encoder._layer_sizes),
        "encoder__n_layers": len(model.encoder._layer_sizes),
        "encoder__activation_function": model.encoder._activation_function,
        "encoder__dropout": model.encoder.dropout,
        "decoder__layer_sizes": str(model.decoder._layer_sizes),
        "decoder__n_layers": len(model.decoder._layer_sizes),
        "decoder__activation_function": model.decoder._activation_function,
        "decoder__dropout": model.decoder.dropout,
    }
