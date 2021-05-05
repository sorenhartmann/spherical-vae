from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import KNeighborsClassifier
from src.experiments.mocap import get_experiment_data, model_args
from src.models.svae import SphericalVAE
from src.models.vae import VariationalAutoencoder

run_dir = Path(__file__).parents[2] / "runs"

if __name__ == "__main__":

    experiments = ["run-walk", "swimming", "dancing", "walk-walk"]

    columns = pd.MultiIndex.from_tuples([['KNN', 'VAE'], ['KNN', 'S-VAE'], ['LL', 'VAE'], ['LL', 'S-VAE'], ['ELBO', 'VAE'], ['ELBO', 'S-VAE']], names=["Estimate", "Model"])
    results = pd.DataFrame(columns = columns, index = experiments)

    for experiment in experiments:

        # Training data
        X_train, classes_train, obs_labels_train = get_experiment_data(experiment, test=False)
        sorted_obs_labels_train = pd.Series(obs_labels_train).str.split(":", expand=True).astype({0:str, 1:int}).sort_values([0, 1])
        trial_ids_train = sorted_obs_labels_train[0]
        sorted_order_train = sorted_obs_labels_train.index

        X_train = X_train[sorted_order_train, :]
        classes_train = np.array(classes_train)[sorted_order_train]

        # Test data
        X_test, classes_test, obs_labels_test = get_experiment_data(experiment, test=True)
        sorted_obs_labels_test = pd.Series(obs_labels_test).str.split(":", expand=True).astype({0:str, 1:int}).sort_values([0, 1])
        trial_ids_test = sorted_obs_labels_test[0]
        sorted_order_test = sorted_obs_labels_test.index

        X_test = X_test[sorted_order_test, :]
        classes_test = np.array(classes_test)[sorted_order_test]

        state_dict_path = run_dir / experiment / "best_vae.pt"
        
        if state_dict_path.exists():
            vae = VariationalAutoencoder(latent_dim=2, **model_args)
            vae_state_dict = torch.load(
                state_dict_path, map_location=torch.device("cpu")
            )
            vae.load_state_dict(vae_state_dict)

            Z_train = vae(X_train)["z"].detach().numpy()
            Z_test = vae(X_test)["z"].detach().numpy()
            knn_vae = KNeighborsClassifier(n_neighbors=3)
            knn_vae.fit(Z_train, classes_train)
            results.loc[experiment, ('KNN', 'VAE')] = knn_vae.score(Z_test, classes_test)

            results.loc[experiment, ('ELBO', 'VAE')] = -vae.get_loss(X_test).detach().numpy()
            results.loc[experiment, ('LL', 'VAE')] = vae.log_likelihood(X_test)['average_log_like'].detach().numpy()

            #print(f"VAE for {experiment}: knn score: {score}, elbo: {elbo}, log likelihood: {log_like}")

        state_dict_path = run_dir / experiment / "best_svae.pt"
        
        if state_dict_path.exists():
            svae = SphericalVAE(latent_dim=3, **model_args)
            svae_state_dict = torch.load(
                state_dict_path, map_location=torch.device("cpu")
            )
            svae.load_state_dict(svae_state_dict)

            Z_train = svae(X_train)["z"].detach().numpy()
            Z_test = svae(X_test)["z"].detach().numpy()
            knn_svae = KNeighborsClassifier(n_neighbors=3)
            knn_svae.fit(Z_train, classes_train)
            results.loc[experiment, ('KNN', 'S-VAE')] = knn_svae.score(Z_test, classes_test)

            results.loc[experiment, ('ELBO', 'S-VAE')] = -svae.get_loss(X_test).detach().numpy()
            results.loc[experiment, ('LL', 'S-VAE')] = svae.log_likelihood(X_test)['average_log_like'].detach().numpy()


            #print(f"SVAE for {experiment}: knn score: {score}, elbo: {elbo}, log likelihood: {log_like}")
        

    results.to_latex()

    
    