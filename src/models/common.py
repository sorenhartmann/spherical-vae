from torch.nn import Module, Sequential, Linear, ReLU


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
            Linear(in_features=100, out_features=out_features)
        )

    def forward(self, x):
        
        return self.ffnn(x)
