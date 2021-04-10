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

        trial_data = {}
        with zipfile.ZipFile(self.raw_file) as zf:
            subject_dir = rf"all_asfamc/subjects/{self.subject}/"
            files = [f for f in zf.filelist if f.filename.startswith(subject_dir)]
            files.sort(key=lambda x: x.filename)
            for file_ in files:
                if file_.filename.endswith(".amc"):
                    with zf.open(file_) as f:
                        data = process_amc(io.BytesIO(f.read()))
                        trial_name = file_.filename[len(subject_dir) : -4]
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


def process_amc(file_contents: io.BytesIO):

    raw = pd.read_csv(file_contents, header=None)
    is_numeric = raw[0].str.isnumeric()
    indices = raw[0].loc[is_numeric]

    raw.loc[is_numeric, "index"] = indices.astype(float)
    raw["index"] = raw["index"].ffill()

    first_observations = raw.loc[raw["index"] == 1].iloc[1:]
    field_names = [
        f"{obs[0]}:{i}"
        for obs in first_observations[0].str.split(" ")
        for i in range(len(obs) - 1)
    ]

    observations = []
    for i, group in raw[~is_numeric].groupby("index", group_keys=False):
        # And now, for the tricky bit
        observations.append(
            group[0].str.split(" ", n=1, expand=True)[1].str.split(" ").explode().values
        )

    data = pd.DataFrame.from_records(observations, columns=field_names,)
    return data.astype(float)


if __name__ == "__main__":

    tmp = MotionCaptureDataset(subject="07")
