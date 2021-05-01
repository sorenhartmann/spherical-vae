import plotly
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
import re
from sklearn.preprocessing import LabelEncoder
sns.set_theme()

run_dir = Path(__file__).parents[2] / "runs"


def spherical_coordiate(x, y, z):

    lambda_ = np.arccos(z) * 2
    phi = np.arctan2(y, x)

    return lambda_, phi


def get_test_data(experiment_name):
    if experiment_name == "run-walk":
        run_data = MotionCaptureDataset("09", test=True)
        walk_data = MotionCaptureDataset("08", test=True)
        X = torch.cat([run_data.X, walk_data.X])
        labels = ["Run"] * len(run_data) + ["Walk"] * len(walk_data)
        return X, labels
    elif experiment_name == "dancing":
        salsa_data = MotionCaptureDataset("60", test=True)
        indian_data = MotionCaptureDataset("94", test=True)
        X = torch.cat([salsa_data.X, indian_data.X])
        labels = ["Salsa"] * len(salsa_data) + ["Indian Dance"] * len(indian_data)
        return X, labels
    elif experiment_name == "swimming":
        swim_data = MotionCaptureDataset("126", test=True)
        num_to_stroke = {
            1: "Back Stroke",
            2: "Back Stroke",
            3: "Breast Stroke",
            4: "Breast Stroke",
            5: "Breast Stroke",
            6: "Fly Stroke",
            7: "Fly Stroke",
            8: "Fly Stroke",
            9: "Fly Stroke",
            10: "Free Style",
            11: "Free Style",
            12: "Free Style",
            13: "Motorcycle",
            14: "Range of Motion",
        }
        pattern = re.compile("_(\d+):") 
        trial_numbers = [int(pattern.search(s).group(1)) for s in swim_data.labels]
        labels = [num_to_stroke[i] for i in trial_numbers]
        return swim_data.X,  labels

if __name__ == "__main__":

    experiments = ["run-walk", "swimming", "dancing"]

    for experiment in experiments:

        X, labels = get_test_data(experiment)

        vae = VariationalAutoencoder(latent_dim=2, **model_args)

        state_dict_path = run_dir / experiment / "best_vae.pt"
        
        if state_dict_path.exists():

            vae_state_dict = torch.load(
                state_dict_path, map_location=torch.device("cpu")
            )
            vae.load_state_dict(vae_state_dict)

            Z = vae(X)["z"]
            x, y = Z.detach().numpy().T
            plt.figure()
            plt.title(f"{experiment}-vae")
            sns.scatterplot(x=x, y=y, hue=labels)
            plt.axis("equal")

        state_dict_path = run_dir / experiment / "best_svae.pt"
        
        if state_dict_path.exists():
            print(state_dict_path)

            svae = SphericalVAE(latent_dim=3, **model_args)
            svae_state_dict = torch.load(
                state_dict_path, map_location=torch.device("cpu")
            )
            svae.load_state_dict(svae_state_dict)
            Z = svae(X)["z"]
            classes = LabelEncoder().fit_transform(labels)
            x, y, z = Z.detach().numpy().T
            fig = go.Figure()
            plot_3d(x, y, z, fig=fig, classes=classes)
            fig.show()

  