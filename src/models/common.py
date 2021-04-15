import torch
from torch.nn import Module, Sequential, Linear, ReLU
from torch.utils.data.dataset import random_split
from tqdm import trange

class Encoder(Module):
    def __init__(self, in_features, out_features):

        super().__init__()

        self.ffnn = Sequential(
            Linear(in_features=in_features, out_features=100),
            ReLU(),
            Linear(in_features=100, out_features=100),
            ReLU(),
            Linear(in_features=100, out_features=out_features),
        )

    def forward(self, x):

        return self.ffnn(x)


class Decoder(Module):
    def __init__(self, in_features, out_features):

        super().__init__()

        self.ffnn = Sequential(
            Linear(in_features=in_features, out_features=100),
            ReLU(),
            Linear(in_features=100, out_features=100),
            ReLU(),
            Linear(in_features=100, out_features=out_features),
        )

    def forward(self, x):

        return self.ffnn(x)

def train_model(
    model,
    dataset,
    train_size=0.8,
    n_epochs=100,
    batch_size=16,
    lr=1e-3,
    checkpoint_path=None,
):

    assert (
        hasattr(model, "get_loss") and callable(model.get_loss),
        "Model needs to have implemented a .get_loss method",
    )

    torch.manual_seed(123)

    train_size = int(train_size * len(dataset))
    validation_size = len(dataset) - train_size

    train_dataset, validation_dataset = random_split(
        dataset, [train_size, validation_size]
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=batch_size
    )

    epoch_train_loss = [None] * len(train_loader)
    epoch_validation_loss = [None] * len(validation_loader)

    train_loss = [None] * n_epochs
    validation_loss = [None] * n_epochs

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in trange(n_epochs):

        model.train()
        for i, batch in enumerate(train_loader):

            # Forward pass
            loss = model.get_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss[i] = loss.item()

        # Validating model
        model.eval()
        with torch.no_grad():

            for i, batch in enumerate(validation_loader):
                # Forward pass
                loss = model.get_loss(batch)
                epoch_validation_loss[i] = loss.item()

        train_loss[epoch] = sum(epoch_train_loss) / len(epoch_train_loss)
        validation_loss[epoch] = sum(epoch_validation_loss) / len(epoch_validation_loss)

        # print(f"train loss: {train_loss[epoch]:.4f}")
        # print(f"validation loss: {validation_loss[epoch]:.4f}")

    if checkpoint_path is not None:
        torch.save(model.state_dict(), checkpoint_path)

    return train_loss, validation_loss