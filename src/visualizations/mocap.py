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
import pandas as pd

sns.set_theme("paper", style="whitegrid")

run_dir = Path(__file__).parents[2] / "runs"
pattern = re.compile("_(\d+):")

def get_test_data(experiment_name):
    if experiment_name == "run-walk":
        run_data = MotionCaptureDataset("09", test=True)
        walk_data = MotionCaptureDataset("08", test=True)

        X = torch.cat([run_data.X, walk_data.X])
        classes = ["Run"] * len(run_data) + ["Walk"] * len(walk_data)
        obs_labels = run_data.labels + walk_data.labels

        return X, classes, obs_labels
    elif experiment_name == "dancing":
        salsa_data = MotionCaptureDataset("60", test=True)
        indian_data = MotionCaptureDataset("94", test=True)
        X = torch.cat([salsa_data.X, indian_data.X])
        classes = ["Salsa"] * len(salsa_data) + ["Indian Dance"] * len(indian_data)
        obs_labels = salsa_data.labels + indian_data.labels
        return X, classes, obs_labels

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
        trial_numbers = [int(pattern.search(s).group(1)) for s in swim_data.labels]
        classes = [num_to_stroke[i] for i in trial_numbers]

        return swim_data.X, classes, swim_data.labels

    elif experiment_name == "walk-walk":
        walk_1_data = MotionCaptureDataset("07")
        walk_2_data = MotionCaptureDataset("08")
        X = torch.cat([walk_1_data.X, walk_2_data.X])
        classes = ["Walk 1"] * len(walk_1_data) + ["Walk 2"] * len(walk_2_data)

        obs_labels = walk_1_data.labels + walk_2_data.labels

        return X, classes, obs_labels


def spherical_coordinates(x, y, z):

    latitude = np.arcsin(z)
    longitude = np.arctan2(y, x)

    return latitude, longitude


def hammer_atioff(latitude, longitude):

    z = np.sqrt(1 + np.cos(latitude) * np.cos(longitude / 2))
    x = 2 * np.cos(latitude) * np.sin(longitude / 2) / z
    y = np.sin(latitude) / z

    return x, y


if __name__ == "__main__":

    experiments = ["run-walk", "swimming", "dancing", "walk-walk"]

    for experiment in experiments:

        plot_data = pd.DataFrame(columns=["x", "y", "Label", "Model"])

        X_, classes_, obs_labels_ = get_test_data(experiment)
        sorted_obs_labels = pd.Series(obs_labels_).str.split(":", expand=True).astype({0:str, 1:int}).sort_values([0, 1])
        trial_ids = sorted_obs_labels[0]
        sorted_order = sorted_obs_labels.index

        X = X_[sorted_order, :]
        classes  =np.array(classes_)[sorted_order]

        trials_to_plot = np.random.default_rng(seed=42).choice(trial_ids.unique(), 3, replace=False)
        plot_mask = trial_ids.isin(trials_to_plot)

        state_dict_path = run_dir / experiment / "best_vae.pt"

        if state_dict_path.exists():
            vae = VariationalAutoencoder(latent_dim=2, **model_args)
            vae_state_dict = torch.load(
                state_dict_path, map_location=torch.device("cpu")
            )
            vae.load_state_dict(vae_state_dict)

            Z = vae(X)["z"]
            x, y = Z.detach().numpy().T

            plt.figure()
            sns.scatterplot(x=x, y=y, hue=classes, marker=".", edgecolor="none")
            plt.axis("equal")

            trials_to_plot

            plt.figure()
            sns.lineplot(x=x, y=y, hue=classes, sort=False, units=trial_ids, estimator=None)
            plt.axis("equal")


        state_dict_path = run_dir / experiment / "best_svae.pt"
        if state_dict_path.exists():
            svae = SphericalVAE(latent_dim=3, **model_args)
            svae_state_dict = torch.load(
                state_dict_path, map_location=torch.device("cpu")
            )
            svae.load_state_dict(svae_state_dict)
            Z = svae(X)["z"]
            x, y, z = Z.detach().numpy().T
            x_, y_ = hammer_atioff(*spherical_coordinates(x, y, z))
            
            plt.figure()
            sns.scatterplot(x=x_, y=y_, hue=classes, marker=".", edgecolor="none")
            plt.axis("equal")

            has_jumped = np.append(np.abs(np.diff(x_)) > 1.5, False)
            run_id = 0
            runs = []
            for jumped in has_jumped:
                runs.append(run_id)
                if jumped:
                    run_id += 1
            units = [f"{trial_id}_{run_id}" for trial_id, run_id in zip(trial_ids, runs)]
            
            plt.figure()
            sns.lineplot(x=x_, y=y_, hue=classes, units=units, estimator=None, sort=False)


        