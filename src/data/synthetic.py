import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from pathlib import Path


def gen_noisy_synth_data_s1(nsamples=100, nclasses=3, new_dim=20):

    # getting synhetic s1 data
    coords, target = gen_synth_data_s1(nsamples, nclasses, plot_cartesian)

    # multiplicative and additive random noise
    noisy_coords = coords @ np.random.normal(
        size=(np.shape(coords)[1], new_dim)
    ) + np.random.normal(size=(np.shape(coords)[0], new_dim))

    return (noisy_coords, target)


def gen_synth_data_s1(nsamples=100, nclasses=3, plot_cartesian=True):
    """
    a function creating synthetic data on s1
    input: @nsamples: the number of samples in each class, integer
        @n_classes: the number of classes, integer
        @plot_cartesian: display a plot of s1 with the points, boolean
    output: @samples_cartesian: cartesian coordinates in r2 of the samples on s1.
                            dimension is (nsamples, 2, nclasses)
    """

    # drawing means between -pi and pi
    means = np.random.uniform(low=-np.pi, high=np.pi, size=nclasses)
    scales = np.random.uniform(low=0.1, high=1, size=nclasses)

    # drawing angles from normal distribution
    thetas = [
        np.random.normal(loc=means[i], scale=scales[i], size=nsamples)
        for i in range(nclasses)
    ]

    # changing to polar coordinates
    samples_cartesian = [
        np.transpose(
            np.array(
                [np.cos(thetas[i]), np.sin(thetas[i]), np.ones(len(thetas[i])) * i]
            )
        )
        for i in range(nclasses)
    ]  # size: samples, 2, nclasses)
    samples_cartesian = np.concatenate(samples_cartesian)

    coords = samples_cartesian[:, 0:2]
    target = samples_cartesian[:, 2]

    if plot_cartesian:
        # plotting samples
        plt.figure(figsize=(5, 5))
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        plt.scatter(
            x=samples_cartesian[:, 0],
            y=samples_cartesian[:, 1],
            c=samples_cartesian[:, 2],
            cmap="tab10",
        )

    return (coords, target)


def gen_synth_data_s2(nclasses=3, nsamples=200):

    # drawing means and sd for theta (inclination) between 0 and pi
    means_theta = np.random.uniform(low=0, high=np.pi, size=nclasses)
    scales_theta = np.random.uniform(low=0.1, high=1, size=nclasses)

    # drawing means and sd for phi (azimuth) between -pi and pi
    means_phi = np.random.uniform(low=0, high=2 * np.pi, size=nclasses)
    scales_phi = np.random.uniform(low=0.1, high=1, size=nclasses)

    # drawing angles from normal distribution
    thetas = [
        np.random.normal(loc=means_theta[i], scale=scales_theta[i], size=nsamples)
        for i in range(nclasses)
    ]
    phis = [
        np.random.normal(loc=means_phi[i], scale=scales_phi[i], size=nsamples)
        for i in range(nclasses)
    ]

    xcoor = lambda theta, phi: np.sin(theta) * np.cos(phi)
    ycoor = lambda theta, phi: np.sin(theta) * np.sin(phi)
    zcoor = lambda theta: np.cos(theta)

    # changing to polar coordinates
    samples_cartesian = [
        np.transpose(
            np.array(
                [
                    xcoor(thetas[i], phis[i]),
                    ycoor(thetas[i], phis[i]),
                    zcoor(thetas[i]),
                    np.ones(len(thetas[i])) * i,
                ]
            )
        )
        for i in range(nclasses)
    ]  # size: samples, [x,y,z,t], nclasses)
    samples_cartesian = np.concatenate(samples_cartesian)

    coords = samples_cartesian[:, 0:3]
    target = samples_cartesian[:, 3]

    return coords, target


def gen_noisy_synth_data_s2(nsamples=200, nclasses=3, new_dim=50):

    # getting synhetic s1 data
    coords, target = gen_synth_data_s2(nsamples=nsamples, nclasses=nclasses)

    # multiplicative and additive random noise
    noisy_coords = coords @ np.random.normal(
        size=(np.shape(coords)[1], new_dim)
    ) + np.random.normal(size=(np.shape(coords)[0], new_dim))

    return (coords, noisy_coords, target)


class SyntheticS2(torch.utils.data.Dataset):

    n_test = 500
    n_train = 1000
    n_classes = 3
    n_features = 50
    seed = 60220

    def __init__(self, test=False):

        self.test = test

        file_name = "test.pt" if test else "train.pt"
        self.file_dir = Path(__file__).parent / ".." / ".." / "data" / "synthetic_s2"

        try:
            data = torch.load(self.file_dir / file_name)
        except FileNotFoundError:
            data = self._generate_data()

        self.X_latent = data["X_latent"]
        self.X = data["X"]
        self.y = data["y"]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index, :]

    def _generate_data(self):

        np.random.seed(self.seed)

        X_latent, X, y = gen_noisy_synth_data_s2(
            (self.n_test + self.n_train) // self.n_classes,
            self.n_classes,
            self.n_features,
        )
        X_latent = torch.tensor(X_latent, dtype=torch.double)
        X = torch.tensor(X, dtype=torch.double)
        y = torch.tensor(y, dtype=torch.double)

        (
            X_latent_train,
            X_latent_test,
            X_train,
            X_test,
            y_train,
            y_test,
        ) = train_test_split(X_latent, X, y, train_size=self.n_train)

        train = {
            "X_latent": X_latent_train,
            "X": X_train,
            "y": y_train,
        }

        test = {
            "X_latent": X_latent_test,
            "X": X_test,
            "y": y_test,
        }

        if not self.file_dir.exists():
            self.file_dir.mkdir()

        torch.save(train, self.file_dir / "train.pt")
        torch.save(test, self.file_dir / "test.pt")

        data = test if self.test else train
        return data


if __name__ == "__main__":

    synthetic_s2 = SyntheticS2()
    synthetic_s2_test = SyntheticS2(test=True)

    print(len(synthetic_s2))
    print(len(synthetic_s2_test))