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
from math import floor, log
from sklearn.metrics import roc_auc_score

def spherical_coordiate(x, y, z):
    lambda_ = np.arccos(z) * 2 
    phi = np.arctan2(y, x) 
    return lambda_, phi

def in_dist_samples(experiment_name, n_samples= 100, seed = 40521):
    torch.manual_seed(seed)
    
    X, _, _ = get_test_data(experiment_name)

    perm = torch.randperm(X.shape[0])
    idx = perm[:n_samples]
    samples = X[idx]

    return(samples)
    
def out_of_dist_samples(n_samples = 100, seed = 40521):
    torch.manual_seed(seed)

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

def get_log_like_and_ELBO(experiment_name, n_samples_in_dist = 100, n_samples_out_dist = 100, n_samples_monte_carlo = 32):
    in_dist_X = in_dist_samples(experiment_name, n_samples=n_samples_in_dist) 
    out_of_dist_X = out_of_dist_samples(n_samples=n_samples_out_dist)

    run_dir = Path(__file__).parents[2] / "runs" / experiment_name

    print(run_dir)

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
    ELBO_vae_in_dist = vae.get_ELBO_per_obs(in_dist_X)
    ELBO_svae_in_dist = svae.get_ELBO_per_obs(in_dist_X)

    log_lik_vae_out_of_dist = vae.log_likelihood(out_of_dist_X, S=n_samples_monte_carlo)["log_like"]
    log_lik_svae_out_of_dist = svae.log_likelihood(out_of_dist_X, S=n_samples_monte_carlo)["log_like"]
    ELBO_vae_out_of_dist = vae.get_ELBO_per_obs(out_of_dist_X)
    ELBO_svae_out_of_dist = svae.get_ELBO_per_obs(out_of_dist_X)

    log_lik_ELBO_dict = {"log_lik_vae_in_dist" : log_lik_vae_in_dist,
                          "log_lik_svae_in_dist": log_lik_svae_in_dist,
                          "ELBO_vae_in_dist": ELBO_vae_in_dist,
                          "ELBO_svae_in_dist": ELBO_svae_in_dist,
                          "log_lik_vae_out_of_dist": log_lik_vae_out_of_dist, 
                          "log_lik_svae_out_of_dist": log_lik_svae_out_of_dist,
                          "ELBO_vae_out_of_dist": ELBO_vae_out_of_dist,
                          "ELBO_svae_out_of_dist": ELBO_svae_out_of_dist                 
                           }

    return log_lik_ELBO_dict

def get_roc_auc_score(log_lik_ELBO_dict, experiment_name):
    log_lik_vae_in_dist = log_lik_ELBO_dict["log_lik_vae_in_dist"]
    log_lik_svae_in_dist = log_lik_ELBO_dict["log_lik_svae_in_dist"]
    ELBO_vae_in_dist = log_lik_ELBO_dict["ELBO_vae_in_dist"]
    ELBO_svae_in_dist = log_lik_ELBO_dict["ELBO_svae_in_dist"]
    log_lik_vae_out_of_dist = log_lik_ELBO_dict["log_lik_vae_out_of_dist"]
    log_lik_svae_out_of_dist = log_lik_ELBO_dict["log_lik_svae_out_of_dist"]
    ELBO_vae_out_of_dist = log_lik_ELBO_dict["ELBO_vae_out_of_dist"]
    ELBO_svae_out_of_dist = log_lik_ELBO_dict["ELBO_svae_out_of_dist"]     
                          
    # ROC curves and AUC for log likelihood experiments
    scores_vae_log_lik = torch.cat((log_lik_vae_out_of_dist,log_lik_vae_in_dist), 0)
    labels_vae_log_lik = torch.cat((torch.zeros(log_lik_vae_out_of_dist.shape[0]), torch.ones(log_lik_vae_in_dist.shape[0])),0)
    
    scores_svae_log_lik = torch.cat((log_lik_svae_out_of_dist, log_lik_svae_in_dist), 0)
    labels_svae_log_lik = torch.cat((torch.zeros(log_lik_svae_out_of_dist.shape[0]), torch.ones(log_lik_svae_in_dist.shape[0])),0)

    roc_auc_svae_log_lik = roc_auc_score(y_true = labels_svae_log_lik, y_score = scores_svae_log_lik)
    roc_auc_vae_log_lik = roc_auc_score(y_true = labels_vae_log_lik, y_score = scores_vae_log_lik)

    # ROC curves and AUC for ELBO experiments
    scores_vae_ELBO = torch.cat((ELBO_vae_out_of_dist,ELBO_vae_in_dist), 0).detach()
    labels_vae_ELBO = torch.cat((torch.zeros(ELBO_vae_out_of_dist.shape[0]), torch.ones(ELBO_vae_in_dist.shape[0])),0)
    
    scores_svae_ELBO = torch.cat((ELBO_svae_out_of_dist,ELBO_svae_in_dist), 0).detach()
    labels_svae_ELBO = torch.cat((torch.zeros(ELBO_svae_out_of_dist.shape[0]), torch.ones(ELBO_svae_in_dist.shape[0])),0)

    roc_auc_svae_ELBO = roc_auc_score(y_true = labels_svae_ELBO, y_score = scores_svae_ELBO)
    roc_auc_vae_ELBO = roc_auc_score(y_true = labels_vae_ELBO, y_score = scores_vae_ELBO)

    print(f"EXperiment: {experiment_name}")
    print(f"roc_auc for Log Likelihood of SVAE:{roc_auc_svae_log_lik}")
    print(f"roc_auc for Log Likelihood of VAE:{roc_auc_vae_log_lik}")
    print(f"roc_auc for ELBO of SVAE:{roc_auc_svae_ELBO}")
    print(f"roc_auc for ELBO of VAE:{roc_auc_vae_ELBO}")

    roc_auc_res = {"log_lik_vae": roc_auc_vae_log_lik, "log_lik_svae": roc_auc_svae_log_lik, "ELBO_vae":roc_auc_vae_ELBO, "ELBO_svae":roc_auc_svae_ELBO}
    return roc_auc_res
    
if __name__ == "__main__":

    log_lik_ELBO_swimming = get_log_like_and_ELBO("swimming", n_samples_in_dist = 200, n_samples_out_dist= 200, n_samples_monte_carlo=125)
    roc_auc_swimming = get_roc_auc_score(log_lik_ELBO_swimming, "swimming")

    log_lik_ELBO_walk_walk = get_log_like_and_ELBO("walk-walk", n_samples_in_dist = 200, n_samples_out_dist= 200, n_samples_monte_carlo=125)
    roc_auc_walk_walk = get_roc_auc_score(log_lik_ELBO_walk_walk, "walk-walk")
    
    log_lik_ELBO_run_walk = get_log_like_and_ELBO("walk-walk", n_samples_in_dist = 200, n_samples_out_dist= 200, n_samples_monte_carlo=125)
    roc_auc_run_walk = get_roc_auc_score(log_lik_ELBO_run_walk, "run-walk")

    log_lik_ELBO_dancing = get_log_like_and_ELBO("dancing", n_samples_in_dist = 200, n_samples_out_dist= 200, n_samples_monte_carlo=125)
    roc_auc_dancing = get_roc_auc_score(log_lik_ELBO_dancing, "dancing")

    colors = ["firebrick", "steelblue"]
    alphas = [1, 0.4]

    fig, ax = plt.subplots()
    for k, log_lik in enumerate([log_lik_ELBO_dancing["log_lik_vae_out_of_dist"], log_lik_ELBO_dancing["log_lik_vae_in_dist"]]):
        sns.histplot(log_lik, 
                     ax = ax,
                     color=colors[k], 
                     stat = "count", 
                     alpha = alphas[k])

    #fig2, ax2 = plt.subplots(figsize = (15,10))
    #for k, log_lik in enumerate([log_lik_svae_out_of_dist, log_lik_svae_in_dist]):
    #            sns.histplot(log_lik, 
    #                 ax = ax2,
    #                 color=colors[k], 
    #                 stat = "count",
    #                 alpha = alphas[k])
