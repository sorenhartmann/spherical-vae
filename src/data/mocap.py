import io
import urllib.request
import torch
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

    def __init__(
        self,
        subject: str,
        test=False,
    ):

        self.test = test
        self.subject = subject
        sub_path = Path("subjects") / self.subject

        self.raw_file = raw_dir / Path(self.raw_url).name
        self.preproccesed_dir = preprocessed_dir / "mocap" /  sub_path

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

        for file_name, buffer in zipped_data.iter_files(subject_dir, ext=".amc"):
            data = process_amc(buffer)
            trial_name = file_name[len(subject_dir)+1:-4]
            trial_data[trial_name] = data

        labels = [
            f"{trial_name}:{i+1}"
            for trial_name, data in trial_data.items()
            for i in range(len(data))
        ]

        X = torch.cat([torch.tensor(data.values) for data in trial_data.values()])

        X_train, X_test, labels_train, labels_test = train_test_split(
            X, labels, random_state=123, train_size=self.train_perc
        )

        train = {
            "X": X_train,
            "labels": labels_train,
        }

        test = {
            "X": X_test,
            "labels": labels_test,
        }

        if not self.preproccesed_dir.exists():
            self.preproccesed_dir.mkdir(parents=True, exist_ok=True)

        torch.save(train, self.preproccesed_dir / "train.pt")
        torch.save(test, self.preproccesed_dir / "test.pt")

        data = test if self.test else train
        return data

    def __len__(self):
        return self.X.shape[-1]

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
            name, *values = line.split(' ')
            observation.extend(float(x) for x in values)
            field_names.extend(f"{name}:{i}" for i in range(len(values)))
        else:
            observations.append(observation)
            observation = []
            break

    # Read remaining lines 
    for line in file_contents: 
        if not line.strip().isnumeric():
            name, *values = line.split(' ')
            observation.extend(float(x) for x in values)
        else:
            observations.append(observation)
            observation = []

    # Save final line
    if len(observation) > 0:
        observations.append(observation)

    data = pd.DataFrame.from_records(observations, columns=field_names,)
    return data

if __name__ == "__main__":

    tmp = MotionCaptureDataset(subject="07")
    tmp.to

