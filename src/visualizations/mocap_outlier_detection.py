import torch
from pathlib import Path
from src.data.mocap import MotionCaptureDataset
from src.experiments.mocap import model_args
from src.visualizations.mocap import get_test_data
import seaborn as sns
from src.models.vae import VariationalAutoencoder
from src.models.svae import SphericalVAE
import matplotlib.pyplot as plt
import numpy as np
from math import floor
from sklearn.metrics import roc_auc_score

run_dir = Path(__file__).parents[2] / "runs" / "swimming"

def spherical_coordiate(x, y, z):
    lambda_ = np.arccos(z) * 2 
    phi = np.arctan2(y, x) 
    return lambda_, phi

def in_dist_samples(experiment_name, n_samples= 100):
    X, _, _ = get_test_data(experiment_name)

    perm = torch.randperm(X.shape[0])
    idx = perm[:n_samples]
    samples = X[idx]

    return(samples)
    
def out_of_dist_samples(n_samples = 100):

    datasets = [
            MotionCaptureDataset("24"),  # nursery rhymes
            MotionCaptureDataset("63"),  # golf swing
            MotionCaptureDataset("140"), # Getting up from ground
            MotionCaptureDataset("115"), # Bending Over
            MotionCaptureDataset("11"),  # Kick Soccer Ball
            MotionCaptureDataset("87"),  # Acrobatics
            MotionCaptureDataset("131")  # Michael Jackson
        ]

    n_samples_per_group = floor(n_samples/len(datasets))

    sample_list = []
    for dataset in datasets:
        perm = torch.randperm(dataset.X.shape[0])
        idx = perm[:n_samples_per_group]
        sample_list.append(dataset.X[idx])
    
    samples = torch.cat(sample_list, 0)
    
    return(samples)

def get_roc_auc_score(experiment_name, n_samples_in_dist = 100, n_samples_out_dist = 100, n_samples_monte_carlo = 32):
    in_dist_X = in_dist_samples(experiment_name, n_samples=n_samples_in_dist) 
    out_of_dist_X = out_of_dist_samples(n_samples=n_samples_out_dist)

    vae = VariationalAutoencoder(latent_dim=2, **model_args)
    vae_state_dict = torch.load(
        run_dir / "best_vae.pt", map_location=torch.device("cpu")
    )
    vae.load_state_dict(vae_state_dict)

    svae = SphericalVAE(latent_dim=3, **model_args)
    svae_state_dict = torch.load(
        run_dir / "best_svae.pt", map_location=torch.device("cpu")
    )
    svae.load_state_dict(svae_state_dict)

    # Outlier detection using log_likelihood 
    log_lik_vae_in_dist = vae.log_likelihood(in_dist_X, S=n_samples_monte_carlo)["log_like"]
    log_lik_svae_in_dist = svae.log_likelihood(in_dist_X, S=n_samples_monte_carlo)["log_like"]

    log_lik_vae_out_of_dist = vae.log_likelihood(out_of_dist_X, S=n_samples_monte_carlo)["log_like"]
    log_lik_svae_out_of_dist = svae.log_likelihood(out_of_dist_X, S=n_samples_monte_carlo)["log_like"]

    colors = ["firebrick", "steelblue"]
    alphas = [1, 0.4]

    fig, ax = plt.subplots()
    for k, log_lik in enumerate([ log_lik_vae_out_of_dist, log_lik_vae_in_dist]):
        sns.histplot(log_lik, 
                     ax = ax,
                     color=colors[k], 
                     stat = "count", 
                     alpha = alphas[k])

    fig2, ax2 = plt.subplots(figsize = (15,10))
    for k, log_lik in enumerate([log_lik_svae_out_of_dist, log_lik_svae_in_dist]):
                sns.histplot(log_lik, 
                     ax = ax2,
                     color=colors[k], 
                     stat = "count",
                     alpha = alphas[k])


    # ROC curves and AUC 
    scores_vae = torch.cat((log_lik_vae_out_of_dist,log_lik_vae_in_dist), 0)
    labels_vae = torch.cat((torch.zeros(log_lik_vae_out_of_dist.shape[0]), torch.ones(log_lik_vae_in_dist.shape[0])),0)

    scores_svae = torch.cat((log_lik_svae_out_of_dist,log_lik_vae_in_dist), 0)
    labels_svae = torch.cat((torch.zeros(log_lik_svae_out_of_dist.shape[0]), torch.ones(log_lik_vae_in_dist.shape[0])),0)

    roc_auc_svae = roc_auc_score(y_true = labels_svae, y_score = scores_svae)
    roc_auc_vae = roc_auc_score(y_true = labels_vae, y_score = scores_vae)

    print(f"EXperiment: {experiment_name}")
    print(f"roc_auc for SVAE:{roc_auc_svae}")
    print(f"roc_auc for VAE:{roc_auc_vae}")

    return roc_auc_svae, roc_auc_vae
    
if __name__ == "__main__":
    roc_auc_svae_swimming, roc_auc_vae_swimming = get_roc_auc_score("swimming")
    roc_auc_svae_dancing, roc_auc_vae_dancing = get_roc_auc_score("dancing")
    roc_auc_svae_walk_walk, roc_auc_vae_walk_walk = get_roc_auc_score("walk-walk")
    roc_auc_svae_walk_run, roc_auc_vae_walk_run = get_roc_auc_score("run-walk")