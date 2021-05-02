import io
import urllib.request
import torch
from torch.utils import data
from torch.utils.data.dataset import Subset
import torchvision
import random
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision.transforms.functional import resize
from tqdm import tqdm
import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data.utils import ZippedData

root_dir = Path(__file__).parents[2]
raw_dir = root_dir / "data" / "raw"
preprocessed_dir = root_dir / "data" / "preprocessed"


def _hook(t):
    last_b = [0]

    def update_to(b=1, bsize=1, tsize=None):
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return update_to


class MotionCaptureDataset(torch.utils.data.Dataset):

    train_perc = 0.7
    raw_url = "http://mocap.cs.cmu.edu/allasfamc.zip"
    name = "mocap"

    def __init__(
        self,
        subject: str,
        test=False,
    ):

        self.test = test
        self.subject = subject
        sub_path = Path("subjects") / self.subject

        self.raw_file = raw_dir / Path(self.raw_url).name
        self.preproccesed_dir = preprocessed_dir / "mocap" / sub_path

        file_name = "test.pt" if test else "train.pt"
        try:
            data = torch.load(self.preproccesed_dir / file_name)
        except FileNotFoundError:
            data = self._load_raw_data()

        self.X = data["X"]
        self.labels = data["labels"]
        self.n_features = self.X.shape[-1]

    def to(self, device):
        self.X = self.X.to(device)

    def _download_raw_data(self):

        self.raw_file.with_suffix(".loading").touch()
        with tqdm(desc="Loading data set", unit="b") as t:
            hook = _hook(t)
            urllib.request.urlretrieve(self.raw_url, self.raw_file, reporthook=hook)
        self.raw_file.with_suffix(".loading").unlink()

    def _load_raw_data(self):

        if not self.raw_file.exists() or self.raw_file.with_suffix(".loading").exists():
            print("Raw data not found, downloading data set...")
            self._download_raw_data()

        zipped_data = ZippedData(self.raw_file)
        subject_dir = f"all_asfamc/subjects/{self.subject}"
        trial_data = {}

        train_data = {"X": [], "labels": []}
        test_data = {"X": [], "labels": []}

        for file_name, buffer in zipped_data.iter_files(subject_dir, ext=".amc"):

            data = process_amc(buffer)
            trial_name = file_name[len(subject_dir) + 1 : -4]
            trial_data[trial_name] = data

            X = torch.tensor(data.values)
            labels = [f"{trial_name}:{i+1}" for i in range(len(X))]
            n_train = int(self.train_perc * len(X))

            train_data["X"].append(X[:n_train, :])
            test_data["X"].append(X[n_train:, :])

            train_data["labels"].append(labels[:n_train])
            test_data["labels"].append(labels[n_train:])

        train = {
            "X": torch.cat(train_data["X"]),
            "labels": sum(train_data["labels"], []),
        }

        test = {
            "X": torch.cat(test_data["X"]),
            "labels": sum(test_data["labels"], []),
        }

        if not self.preproccesed_dir.exists():
            self.preproccesed_dir.mkdir(parents=True, exist_ok=True)

        torch.save(train, self.preproccesed_dir / "train.pt")
        torch.save(test, self.preproccesed_dir / "test.pt")

        data = test if self.test else train
        return data

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index, :]


def process_amc(file_contents: io.BytesIO):

    for initial_line in file_contents:
        initial_line = initial_line.strip()
        if initial_line.strip().isnumeric():
            break

    observations = []
    field_names = []

    # Read first observation
    observation = []
    for line in file_contents:
        if not line.strip().isnumeric():
            name, *values = line.split(" ")
            observation.extend(float(x) for x in values)
            field_names.extend(f"{name}:{i}" for i in range(len(values)))
        else:
            observations.append(observation)
            observation = []
            break

    # Read remaining lines
    for line in file_contents:
        if not line.strip().isnumeric():
            name, *values = line.split(" ")
            observation.extend(float(x) for x in values)
        else:
            observations.append(observation)
            observation = []

    # Save final line
    if len(observation) > 0:
        observations.append(observation)

    data = pd.DataFrame.from_records(
        observations,
        columns=field_names,
    )
    return data


def split_time_series(dataset, split_perc=0.7):

    observations = (
        pd.Series(dataset.labels)
        .str.split(":", expand=True)
        .astype({0: str, 1: int})
        .rename(columns={0: "trial_id", 1: "time_step"})
    )
    counts = observations.groupby("trial_id").count()
    with_total = pd.merge(observations, counts, how="inner", on="trial_id", suffixes=["", "_total"])

    is_first = with_total["time_step"] < with_total["time_step_total"] * split_perc

    idx_first = observations.index[is_first].values
    idx_last = observations.index[~is_first].values

    return Subset(dataset, idx_first), Subset(dataset, idx_last)