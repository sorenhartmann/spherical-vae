from math import floor
from pathlib import Path
from src.models.common import ModelTrainer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from src.experiments.mocap import get_test_data, model_args
from src.models.svae import SphericalVAE
from src.models.vae import VariationalAutoencoder
from src.visualizations.common import hammer_atioff, spherical_coordinates
from src.visualizations.mocap_outlier_detection import (
    out_of_dist_samples,
    in_dist_samples,
    hue_order,
    col_order
)

sns.set_theme("talk", style="whitegrid")

run_dir = Path(__file__).parents[2] / "runs"
overleaf_dir = Path(__file__).resolve().parents[2] / "overleaf" / "figures"

experiments = [
    ("Swimming", "swimming"),
    ("Walking", "walk-walk"),
    ("Run/walk", "run-walk"),
    ("Dancing", "dancing"),
]

def test_set_plots(test_df):

    for exp_name, exp_id in experiments:

        sub_df = test_df[test_df["Experiment"] == exp_name].copy()
        class_order = sorted(sub_df["Class"].unique())

        fig = sns.relplot(
            data=sub_df.sample(frac=1.0),
            x="x",
            y="y",
            hue="Class",
            hue_order=class_order,
            col="Model",
            col_order=col_order,
            edgecolor="none",
            marker=".",
            facet_kws={
                "gridspec_kws": {"width_ratios": [1, 2]},
                "subplot_kws": {"aspect": "equal"},
                "sharex": False,
                "sharey": False,
            },
        )
        fig.savefig(overleaf_dir / f"{exp_id}_latent.pdf")

        svae_df = sub_df["x"][sub_df["Model"] == "SVAE"]
        has_jumped = svae_df.diff().abs() > 1
        run_ids = has_jumped.cumsum()

        sub_df.loc[:, "units"] = sub_df["Trial ID"]
        sub_df.loc[svae_df.index, "units"] = (
            sub_df.loc[svae_df.index, "units"] + "_" + run_ids.astype(str)
        )

        fig = sns.relplot(
            data=sub_df,
            x="x",
            y="y",
            hue="Class",
            hue_order=class_order,
            units="units",
            estimator=None,
            col="Model",
            col_order=col_order,
            kind="line",
            sort=False,
            facet_kws={
                "gridspec_kws": {"width_ratios": [1, 2]},
                "subplot_kws": {"aspect": "equal"},
                "sharex": False,
                "sharey": False,
            },
        )
        fig.savefig(overleaf_dir / f"{exp_id}_latent_w_lines.pdf")


def outlier_plots(outlier_df):

    for exp_name, exp_id in experiments:

        sub_df = outlier_df[outlier_df["Experiment"] == exp_name].copy()

        fig = sns.relplot(
            data=sub_df.sample(frac=1.0),
            x="x",
            y="y",
            hue="Data",
            col="Model",
            col_order=col_order,
            hue_order=hue_order,
            edgecolor="none",
            marker=".",
            palette=sns.color_palette("husl", 2),
            facet_kws={
                "gridspec_kws": {"width_ratios": [1, 2]},
                "subplot_kws": {"aspect": "equal"},
                "sharex": False,
                "sharey": False,
            },
        )
        fig.savefig(overleaf_dir / f"{exp_id}_latent_outliers.pdf")


if __name__ == "__main__":

    test_df = pd.DataFrame(
        columns=["x", "y", "Model", "Experiment", "Class", "Trial ID"]
    )
    outlier_df = pd.DataFrame(columns=["x", "y", "Model", "Experiment", "Data"])

    for exp_name, exp_id in experiments:

        X_, classes_, obs_labels_ = get_test_data(exp_id)

        sorted_obs_labels = (
            pd.Series(obs_labels_)
            .str.split(":", expand=True)
            .astype({0: str, 1: int})
            .sort_values([0, 1])
        )

        trial_ids = sorted_obs_labels[0]
        sorted_order = sorted_obs_labels.index

        X = X_[sorted_order, :]

        X_in_dist = in_dist_samples(exp_id, 1000)
        X_out_dist = out_of_dist_samples(1000)
        X_w_outliers = torch.cat([X_in_dist, X_out_dist])

        classes = np.array(classes_)[sorted_order]

        state_dict_path = run_dir / exp_id / "best_vae.pt"

        if state_dict_path.exists():

            vae = VariationalAutoencoder(latent_dim=2, **model_args)
            vae_state_dict = torch.load(
                state_dict_path, map_location=torch.device("cpu")
            )
            vae.load_state_dict(vae_state_dict)

            Z = vae(X)["z"]
            x, y = Z.detach().numpy().T

            test_df = test_df.append(
                pd.DataFrame(
                    {
                        "x": x,
                        "y": y,
                        "Model": "VAE",
                        "Experiment": exp_name,
                        "Class": classes,
                        "Trial ID": trial_ids,
                    }
                )
            )

            Z_w_outliers = vae(X_w_outliers)["z"]
            x, y = Z_w_outliers.detach().numpy().T

            outlier_df = outlier_df.append(
                pd.DataFrame(
                    {
                        "x": x,
                        "y": y,
                        "Experiment": exp_name,
                        "Model": "VAE",
                        "Data": ["In dist."] * len(X_in_dist)
                        + ["Out of dist."] * len(X_out_dist),
                    }
                )
            )

        state_dict_path = run_dir / exp_id / "best_svae.pt"

        if state_dict_path.exists():

            svae = SphericalVAE(latent_dim=3, **model_args)
            svae_state_dict = torch.load(
                state_dict_path, map_location=torch.device("cpu")
            )
            svae.load_state_dict(svae_state_dict)
            Z = svae(X)["z"]
            x, y, z = Z.detach().numpy().T
            x_, y_ = hammer_atioff(*spherical_coordinates(x, y, z))

            test_df = test_df.append(
                pd.DataFrame(
                    {
                        "x": x_,
                        "y": y_,
                        "Model": "SVAE",
                        "Experiment": exp_name,
                        "Class": classes,
                        "Trial ID": trial_ids,
                    }
                )
            )

            Z_w_outliers = svae(X_w_outliers)["z"]
            x, y, z = Z_w_outliers.detach().numpy().T
            x_, y_ = hammer_atioff(*spherical_coordinates(x, y, z))

            outlier_df = outlier_df.append(
                pd.DataFrame(
                    {
                        "x": x_,
                        "y": y_,
                        "Experiment": exp_name,
                        "Model": "SVAE",
                        "Data": ["In dist."] * len(X_in_dist)
                        + ["Out of dist."] * len(X_out_dist),
                    }
                )
            )

    test_df.reset_index(inplace=True, drop=True)
    outlier_df.reset_index(inplace=True, drop=True)

    with sns.plotting_context({"axes.grid": False}):
        test_set_plots(test_df)
        outlier_plots(outlier_df)

        # plt.figure()
        # Z_in_dist = vae(X_in_dist)["z"].detach()
        # Z_out_dist = vae(X_out_dist)["z"].detach()
        # sns.scatterplot(*Z_in_dist.T)
        # sns.scatterplot(*Z_out_dist.T)
