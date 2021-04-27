import os
from datetime import datetime
from pathlib import Path

import optuna
import torch
from torch.nn import Dropout, Linear, Module, Sequential
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
from tqdm import trange


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
        trial=None,
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

        self.best_params = None
        self.best_loss = None
        
        self.init_tb_writer(tb_dir=tb_dir, tb_label=tb_label )
        self.trial = trial


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

        self.epoch_train_loss = [None] * len(self.train_loader)
        self.epoch_validation_loss = [None] * len(self.validation_loader)
        self.epoch_kl_divergence = [None] * len(self.validation_loader)

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

        # Perfom loops
        self.train_loop(epoch)
        self.validation_loop(epoch)

        ## Calculate statistics
        self.train_loss[epoch] = sum(self.epoch_train_loss) / len(self.epoch_train_loss)
        self.validation_loss[epoch] = sum(self.epoch_validation_loss) / len(self.epoch_validation_loss)
        self.kl_divergence[epoch] = sum(self.epoch_kl_divergence) / len(self.epoch_kl_divergence)

        ## Record best model
        if self.best_loss is None or self.validation_loss[epoch] < self.best_loss:
            self.best_loss = self.validation_loss[epoch]
            self.best_params = self.model.state_dict()
        
        ## Log to tensorboard
        if self.tb_writer is not None:
            self.tb_writer.add_scalar("Loss/Train", self.train_loss[epoch], epoch)
            self.tb_writer.add_scalar("Loss/Validation", self.validation_loss[epoch], epoch)
            self.tb_writer.add_scalar("Average KL-term", self.kl_divergence[epoch], epoch)

        if self.trial is not None:
            self.trial.report(self.validation_loss[epoch], epoch)
            if self.trial.should_prune():
                raise optuna.TrialPruned()

    def after_training(self):

        ## Log hyper parameters to tensorboard
        if self.tb_writer is not None:
            hparams = {"lr": self.lr, "batch_size": self.batch_size, "beta_0": self.beta_function.beta_0}
            hparams.update(get_hparams(self.model))
            metrics = {"neg_ELBO": min(self.validation_loss)}
            self.tb_writer.add_hparams(hparams, metrics)
            self.tb_writer.close()

        if self.checkpoint_path is not None:
            Path(self.checkpoint_path).parent.mkdir(exist_ok=True, parents=True)
            torch.save(self.model.state_dict(), self.checkpoint_path)


def get_hparams(model):
    parameters = {
        "latent_dim": lambda x: x.latent_dim,
        "encoder_layer_sizes": lambda x: str(x.encoder._layer_sizes),
        "encoder_n_layers": lambda x: len(x.encoder._layer_sizes),
        "encoder_activation_function": lambda x: x.encoder._activation_function,
        "encoder_dropout": lambda x: x.encoder.dropout,
        "decoder_layer_sizes": lambda x: str(x.decoder._layer_sizes),
        "decoder_n_layers": lambda x: len(x.decoder._layer_sizes),
        "decoder_activation_function": lambda x: x.decoder._activation_function,
        "decoder_dropout": lambda x: x.decoder.dropout,
        ## Conv.
        "encoder_kernel_size" : lambda x: str(x.encoder._kernel_size),
        "encoder_stride" : lambda x: str(x.encoder._stride),
        "encoder_out_channel_size" : lambda x: str(x.encoder._out_channel_size),
        "encoder_padding_size" : lambda x: str(x.encoder._padding_size),
        "encoder_activation_function" : lambda x: x.encoder._activation_function,
        "encoder_dropout2d" : lambda x: x.encoder._dropout2d,
        "encoder_ffnn_layer_size" : lambda x: str(x.encoder._ffnn_layer_size),
        "encoder_ffnn_n_layers" : lambda x: len(x.encoder._ffnn_layer_size),
        "encoder_maxpool_kernel" : lambda x: x.encoder._maxpool_kernel,
        "encoder_maxpool_stride" : lambda x: x.encoder._maxpool_stride,
        "decoder_kernel_size" : lambda x: str(x.decoder._kernel_size),
        "decoder_stride" : lambda x: str(x.decoder._stride),
        "decoder_in_channel_size" : lambda x: str(x.decoder._in_channel_size),
        "decoder_ffnn_layer_size" : lambda x: str(x.decoder._ffnn_layer_size),
        "decoder_fnnn_n_layers" : lambda x: len(x.decoder._ffnn_layer_size),
        "decoder_dropout2d" : lambda x: x.decoder._dropout2d,
    }
    hparams = {}
    for name, getter in parameters.items():
        try:
            hparams[name] = getter(model)
        except AttributeError:
            pass

    return hparams
