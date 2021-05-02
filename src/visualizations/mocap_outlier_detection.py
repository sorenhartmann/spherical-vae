import torch
from pathlib import Path
from src.data.mocap import MotionCaptureDataset
from src.experiments.mocap import model_args
import seaborn as sns
from src.models.vae import VariationalAutoencoder
from src.models.svae import SphericalVAE
from src.utils import plot_3d
from plotly import graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

run_dir = Path(__file__).parents[2] / "runs" / "swimming"

def spherical_coordiate(x, y, z):
    lambda_ = np.arccos(z) * 2 
    phi = np.arctan2(y, x) 
    return lambda_, phi


if __name__ == "__main__":
    # Outlier detection on swimming data 
    # out of dist data is currently Salsa Data

    swim_data = MotionCaptureDataset("126", test=True)
    out_of_dist_data = MotionCaptureDataset("60", test = True)

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

    z_swim_svae = svae(swim_data.X)["z"].detach().numpy()
    z_out_of_dist_svae = svae(out_of_dist_data.X)["z"].detach().numpy()


    # Outlier detection using log_likelihood 
    log_lik_vae_swim = vae.log_likelihood(swim_data.X, S=32)["log_like"]
    log_lik_svae_swim = svae.log_likelihood(swim_data.X, S=32)["log_like"]

    log_lik_vae_out_of_dist = vae.log_likelihood(out_of_dist_data.X, S=32)["log_like"]
    log_lik_svae_out_of_dist = svae.log_likelihood(out_of_dist_data.X, S=32)["log_like"]

    colors = ["firebrick", "steelblue"]
    alphas = [1, 0.4]

    fig, ax = plt.subplots()
    for k, log_lik in enumerate([ log_lik_vae_out_of_dist, log_lik_vae_swim]):
        sns.histplot(log_lik["log_like"], 
                     ax = ax,
                     color=colors[k], 
                     stat = "frequency")

    fig2, ax2 = plt.subplots(figsize = (15,10))
    for k, log_lik in enumerate([log_lik_svae_out_of_dist, log_lik_svae_swim]):
                sns.histplot(log_lik["log_like"], 
                     ax = ax2,
                     color=colors[k], 
                     stat = "count",
                     binwidth=50,
                     alpha = alphas[k])


    # ROC curves and AUC 
    scores_vae = torch.cat((log_lik_vae_out_of_dist,log_lik_vae_swim), 0)
    labels_vae = torch.cat((torch.zeros(log_lik_vae_out_of_dist.shape[0]), torch.ones(log_lik_vae_swim.shape[0])),0)

    scores_svae = torch.cat((log_lik_svae_out_of_dist,log_lik_svae_swim), 0)
    labels_svae = torch.cat((torch.zeros(log_lik_svae_out_of_dist.shape[0]), torch.ones(log_lik_vae_swim.shape[0])),0)

    roc_auc_svae = roc_auc_score(y_true = labels_svae, y_score = scores_svae)
    roc_auc_vae = roc_auc_score(y_true = labels_vae, y_score = scores_vae)