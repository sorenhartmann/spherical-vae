import torch
from pathlib import Path
from src.data.mocap import MotionCaptureDataset
from src.experiments.run_walk import model_args
import seaborn as sns
from src.models.vae import VariationalAutoencoder
from src.models.svae import SphericalVAE
from src.utils import plot_3d
from plotly import graph_objects as go
import matplotlib.pyplot as plt
import numpy as np

run_dir = Path(__file__).parents[2] / "runs" / "run-walk"


def spherical_coordiate(x, y, z):

    lambda_ = np.arccos(z) * 2 
    phi = np.arctan2(y, x) 

    return lambda_, phi


if __name__ == "__main__":

    run_data = MotionCaptureDataset("09", test=True)
    walk_data = MotionCaptureDataset("08", test=True)

    vae = VariationalAutoencoder(latent_dim=2, **model_args)

    vae_state_dict = torch.load(
        run_dir / "best_vae.pt", map_location=torch.device("cpu")
    )
    vae.load_state_dict(vae_state_dict)

    sns.scatterplot(*vae(run_data.X)["z"].detach().numpy().T)
    sns.scatterplot(*vae(walk_data.X)["z"].detach().numpy().T)

    svae = SphericalVAE(latent_dim=3, **model_args)
    svae_state_dict = torch.load(
        run_dir / "best_svae.pt", map_location=torch.device("cpu")
    )
    svae.load_state_dict(svae_state_dict)

    z_run_svae = svae(run_data.X)["z"].detach().numpy()
    z_walk_svae = svae(walk_data.X)["z"].detach().numpy()

    fig = go.Figure()
    plot_3d(*z_run_svae.T, fig=fig)
    plot_3d(*z_walk_svae.T, fig=fig)

    plt.subplot(projection="hammer")
    plt.scatter(*spherical_coordiate(*z_run_svae.T))
    plt.scatter(*spherical_coordiate(*z_walk_svae.T))
    # sns.scatterplot(*hammer_projection(*z_run_svae.T))
    # sns.scatterplot(*hammer_projection(*z_walk_svae.T))
