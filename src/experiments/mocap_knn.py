from src.data.mocap import MotionCaptureDataset
from src.models.vae import VariationalAutoencoder
from src.models.svae import SphericalVAE
from src.experiments.mocap import model_args
from src.visualizations.mocap import get_test_data
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from torch.utils.data import ConcatDataset
from src.utils import plot_3d
from pathlib import Path
import optuna
import torch
import pandas as pd
import numpy as np

run_dir = Path(__file__).parents[2] / "runs"

if __name__ == "__main__":

    experiments = ["run-walk", "swimming", "dancing", "walk-walk"]

    for experiment in experiments:

        X_, classes_, obs_labels_ = get_test_data(experiment)
        sorted_obs_labels = pd.Series(obs_labels_).str.split(":", expand=True).astype({0:str, 1:int}).sort_values([0, 1])
        trial_ids = sorted_obs_labels[0]
        sorted_order = sorted_obs_labels.index

        X = X_[sorted_order, :]
        classes = np.array(classes_)[sorted_order]

        state_dict_path = run_dir / experiment / "best_vae.pt"
        
        if state_dict_path.exists():
            vae = VariationalAutoencoder(latent_dim=2, **model_args)
            vae_state_dict = torch.load(
                state_dict_path, map_location=torch.device("cpu")
            )
            vae.load_state_dict(vae_state_dict)

            Z = vae(X)["z"].detach().numpy()
            knn_vae = KNeighborsClassifier(n_neighbors=3)
            knn_vae.fit(Z, classes)

            print(f"VAE for {experiment}: {cross_val_score(knn_vae, Z, classes)}")

        state_dict_path = run_dir / experiment / "best_svae.pt"
        
        if state_dict_path.exists():
            svae = SphericalVAE(latent_dim=3, **model_args)
            svae_state_dict = torch.load(
                state_dict_path, map_location=torch.device("cpu")
            )
            svae.load_state_dict(svae_state_dict)

            Z = svae(X)["z"].detach().numpy()
            knn_svae = KNeighborsClassifier(n_neighbors=3)
            knn_svae.fit(Z, classes)

            print(f"S-VAE for {experiment}: {cross_val_score(knn_svae, Z, classes)}")


