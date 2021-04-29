from pathlib import Path
from torch.utils.data import random_split
from src.data.mocap import MotionCaptureDataset
from src.models.svae import SphericalVAE
from src.models.vae import VariationalAutoencoder
from src.models.common import ModelTrainer, BetaFunction
from torch.utils.data import ConcatDataset
import torch

cuda = torch.cuda.is_available()
if cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

run_dir = Path(__file__).parents[2] / "runs" / "run-walk"
run_dir.mkdir(exist_ok=True, parents=True)

run_data = MotionCaptureDataset("09")
walk_data = MotionCaptureDataset("08")
assert run_data.n_features == walk_data.n_features
n_features = run_data.n_features

layer_sizes = [700, 750, 700]

latent_dim = 2
model_args = {
        "feature_dim" : n_features,
        "encoder_params" : {
            "layer_sizes": layer_sizes,
            "dropout": 0.1,
            "activation_function": "Tanh",
        },
        "decoder_params" : {
            "layer_sizes": layer_sizes[::-1],
            "dropout": 0.1,
            "activation_function": "Tanh",
        },
}

if __name__ == "__main__":

    torch.manual_seed(123)

    run_data.to(device)
    walk_data.to(device)

    train_split = 0.7

    dataset = ConcatDataset([run_data, walk_data])
    train_size = int(train_split * len(dataset))
    validation_size = len(dataset) - train_size
    train_dataset, validation_dataset = random_split(
        dataset,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(42),
    )

    n_models_to_train = 5
    n_epochs = 1000
    batch_size = 16
    lr = 1e-4
    beta_0 = 0.2

    trainer_args = { 
        "n_epochs" : n_epochs,
        "batch_size" : batch_size,
        "lr" : lr,
        "beta_function" : BetaFunction(beta_0, n_epochs),
    }

    train_args = {
        "train_dataset" : train_dataset,
        "validation_dataset" : validation_dataset,
        "random_state" : None,
        "progress_bar" : True,
    }


    best_svae_loss = None
    best_vae_loss = None

    for i in range(n_models_to_train):

        ## Fit some models, choose best
        vae = VariationalAutoencoder(latent_dim=latent_dim, **model_args)
        vae.to(device)
        mt = ModelTrainer(vae, **trainer_args, tb_dir=run_dir / f"vae_{i}")
        mt.train(**train_args)
        loss = min(mt.validation_loss)
        if best_vae_loss is None or loss < best_vae_loss:
            best_vae_loss = loss
            torch.save(vae.state_dict(), run_dir / "best_vae.pt")

        ## Fit some models, choose best
        svae = SphericalVAE(latent_dim=latent_dim+1, **model_args)
        svae.to(device)
        mt = ModelTrainer(svae, **trainer_args, tb_dir=run_dir / f"svae_{i}")
        mt.train(**train_args)
        loss = min(mt.validation_loss)
        if best_svae_loss is None or loss < best_svae_loss:
            best_svae_loss = loss
            torch.save(svae.state_dict(), run_dir / "best_svae.pt")


   
    
    


