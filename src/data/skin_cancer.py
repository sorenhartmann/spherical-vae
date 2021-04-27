import torch
import torchvision
import random
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision.transforms.functional import resize
from tqdm import tqdm


class SkinCancerDataset(torch.utils.data.Dataset):

    ham_shape = (3, 450, 600)
    train_perc = 0.7

    def __init__(self, subsample=0.1, test=False, image_size=None):
        if image_size is None:
            self.image_size = self.ham_shape[-2:]
        else:
            self.image_size = torch.Size(image_size)

        self.test = test

        self.subsample = subsample

        file_name = "test.pt" if test else "train.pt" # this is not ready yet
        self.file_dir = (
            Path(__file__).parents[2] / "data" / "preprocessed" / "ham10000"
        )

        try:
            data = torch.load(self.file_dir / file_name)
            if data["X"].shape[-2:] != self.image_size:
                data = self._load_raw_data()
        except FileNotFoundError:
            data = self._load_raw_data()

        self.X = data["X"]
        self.image_files = data["image_files"]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index, :, :, :]

    def _load_raw_data(self):

        self.raw_file_dir = (
            Path(__file__).parents[2] / "data" / "raw" / "ham10000"
        )
         
        ham1 = (self.raw_file_dir / "HAM10000_images_part_1")
        ham2 = (self.raw_file_dir / "HAM10000_images_part_1")
        ham_list = list(ham1.iterdir()) + list(ham2.iterdir())

        n_samples = int(len(ham_list)*self.subsample)

        random.seed(270421)
        ham_samples = random.sample(ham_list, k = n_samples) 

        X = torch.zeros((n_samples,) + (3,) + self.image_size)

        for i, imagefile in tqdm(enumerate(ham_samples), total=n_samples):
            if imagefile.suffix != ".jpg":
                continue
            image = torchvision.io.read_image(str(imagefile))
            X[i, :, :, :] = resize(image, size=self.image_size)


        n_train = int(n_samples*self.train_perc)
        
        X_train = X[0:n_train, :, :, :]
        image_files_train = ham_samples[0:n_train]

        X_test = X[n_train:, :, :, :]
        image_files_test = ham_samples[n_train:]

        train = {
            "X": X_train.clone(),
            "image_files": image_files_train,
        }

        test = {
            "X": X_test.clone(),
            "image_files": image_files_test,
        }

        if not self.file_dir.exists():
            self.file_dir.mkdir()

        torch.save(train, self.file_dir / "train.pt")
        torch.save(test, self.file_dir / "test.pt")

        data = test if self.test else train
        return data

    def to(self, device):
        self.X = self.X.to(device)

    def to_dtype(self, device):
        self.X = self.X.to(device)


if __name__ == "__main__": 
 
    tmp = SkinCancerDataset(image_size=[225, 300])
    
    plt.figure(figsize=(8, 8))
    # plt.imshow(image, cmap='gray')