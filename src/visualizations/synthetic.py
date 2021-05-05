from pathlib import Path

import seaborn as sns
import torch
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from src.data import SyntheticS2
from src.models.svae import SphericalVAE
from src.models.vae import VariationalAutoencoder
from src.utils import plot_3d


run_dir = Path(__file__).parents[2] / "runs"
overleaf_dir = Path(__file__).resolve().parents[2] / "overleaf" / "figures"

layer_sizes = [100, 100]
latent_dim = 3
n_features = 50

model_args = {
    "latent_dim": latent_dim,
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

if __name__ == "__main__":

    fig = make_subplots(
        rows=2, 
        cols=2,     
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}],
            [{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=(
            "True latent representation", 
            "PCA", "Ordinary VAE", 
            "Spherical VAE"),
        horizontal_spacing=0,
        vertical_spacing=0.1,
    )


    synthetic_s2 = SyntheticS2(test=True)

    X_latent = synthetic_s2.X_latent
    X = synthetic_s2.X
    classes = synthetic_s2.y

    feature_dim = X.shape[-1]

    svae = SphericalVAE(**model_args)
    svae_state_dict = torch.load(run_dir / "synthetic" / "svae.pt")
    svae.load_state_dict(svae_state_dict)

    vae = VariationalAutoencoder(**model_args)
    vae_state_dict = torch.load(run_dir / "synthetic" / "vae.pt")
    vae.load_state_dict(vae_state_dict)

    plot_3d( *X_latent.T, classes, fig=fig, row=1, col=1)

    pca = PCA(n_components=3)
    X_transformed = pca.fit_transform(X.numpy())
    plot_3d( *X_transformed.T, classes, fig=fig, row=1, col=2)

    output = svae(X)
    plot_3d( *output["z"].T.detach().numpy(), classes, fig=fig, row=2, col=2)

    output = vae(X)
    plot_3d( *output["z"].T.detach().numpy(), classes, fig=fig, row=2, col=1)

    eye = dict(x=-1.25, y=1.25, z=1.25)

    fig.layout.scene.camera.eye = eye
    fig.layout.scene2.camera.eye = eye
    fig.layout.scene3.camera.eye = eye
    fig.layout.scene4.camera.eye = eye
    fig.update_layout(
        height=700,
        width=700,
        showlegend=False,
    )

    if overleaf_dir.exists():
        fig.write_image(str(overleaf_dir / "synthetic_latent.pdf"))
