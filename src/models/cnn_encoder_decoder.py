import torch
import numpy as np
from torch.nn import (
    Module,
    Sequential,
    Linear,
    Conv2d,
    Dropout2d,
    BatchNorm2d,
    MaxPool2d,
    ConvTranspose2d,
)
from torch.nn.modules.dropout import Dropout


def compute_conv_dim(dim_size, kernel_size, padding, stride):
    return int((dim_size - kernel_size + 2 * padding) / stride + 1)

def compute_conv_transpose_out_dim(in_dim, kernel_size, padding, dialation, stride):
    return int((in_dim - 1) * stride - 2 * padding + dialation * (kernel_size - 1) + 1)

def compute_last_kernel_dim(out_dim_size, dim_size):
    # For fixed stride=1, padding = 0, dialation=1
    return int(-dim_size + out_dim_size + 1)

class Encoder(Module):
    def __init__(
        self,
        image_size,
        out_features,
        kernel_size=[3, 5, 3],
        stride=[1, 2, 2],
        out_channel_size=[6, 9, 6],
        padding_size=[1, 2, 1],
        activation_function="Tanh",
        ffnn_layer_size=[100, 1000],
        dropout=None,
        dropout2d=None,
        maxpool_kernel=2,
        maxpool_stride=2,
    ):

        super().__init__()

        self.dropout = dropout
        self.dropout2d = dropout2d

        self._activation_function = activation_function

        ActFunc = getattr(torch.nn.modules.activation, activation_function)

        # CONVOLUTIONAL LAYERS ------------------------------------------------------------------
        convs = []
        channels, height, width = image_size
        for parameters in list(
            zip(out_channel_size, kernel_size, stride, padding_size)
        ):
            convs.append(
                Conv2d(
                    in_channels=channels,
                    out_channels=parameters[0],
                    kernel_size=parameters[1],
                    stride=parameters[2],
                    padding=parameters[3],
                )
            )
            convs.append(BatchNorm2d(parameters[0]))

            if dropout2d is not None:
                convs.append(Dropout2d(dropout2d))

            convs.append(ActFunc())

            height = compute_conv_dim(
                height, parameters[1], parameters[3], parameters[2]
            )
            width = compute_conv_dim(width, parameters[1], parameters[3], parameters[2])
            channels = parameters[0]

        convs.append(MaxPool2d(kernel_size=maxpool_kernel, stride=maxpool_stride))
        height = compute_conv_dim(height, maxpool_kernel, 0, maxpool_stride)
        width = compute_conv_dim(width, maxpool_kernel, 0, maxpool_stride)

        self.CNN = Sequential(*convs)

        self.linear_in_features = channels * height * width
        self.last_im_size = (channels, height, width)

        # FEED FORWARD LAYERS -----------------------------------------------------------------
        if ffnn_layer_size is None:
            ffnn_layer_size = [100, 100]
        self._ffnn_layer_size = ffnn_layer_size

        layers = []
        in_size = self.linear_in_features
        for out_size in ffnn_layer_size:
            layers.append(Linear(in_size, out_size))
            layers.append(ActFunc())
            if dropout is not None:
                layers.append(Dropout(dropout))
            in_size = out_size
        layers.append(Linear(in_size, out_features))

        self.ffnn = Sequential(*layers)

    def forward(self, x):

        x = self.CNN(x)
        x = self.ffnn(x.view(-1, self.linear_in_features))
        return x


class Decoder(Module):
    def __init__(
        self,
        in_features,
        image_size,
        kernel_size=[2, 3, 2],
        stride=[2, 2, 1],
        in_channel_size=[6, 9, 6],
        activation_function="Tanh",
        ffnn_layer_size=[100, 1000],
        dropout=None,
        dropout2d=None,
    ):

        super().__init__()

        ActFunc = getattr(torch.nn.modules.activation, activation_function)

        # TRANSPOSED CONVOLUTIONAL LAYERS ------------------------------------------------------

        convs_reversed = []
        out_channels, out_height, out_width = image_size

        parameters = list(zip(in_channel_size, kernel_size, stride))

        for in_channel_size_, kernel_size_, stride_ in parameters[::-1]:

            if len(convs_reversed) > 0:
                convs_reversed.append(ActFunc())
                if dropout2d is not None:
                    convs_reversed.append(Dropout2d(dropout2d))
                convs_reversed.append(BatchNorm2d(out_channels))

            M_height = out_height - kernel_size_ + stride_
            M_width = out_width - kernel_size_ + stride_
            out_padding_height = M_height % stride_
            out_padding_width = M_width % stride_

            in_height = (M_height - out_padding_height) // stride_
            in_width = (M_width - out_padding_width) // stride_
            
            conv_layer = ConvTranspose2d(
                in_channels=in_channel_size_,
                out_channels=out_channels,
                kernel_size=kernel_size_,
                stride=stride_,
                padding=0,
                output_padding=(out_padding_height, out_padding_width),
            )

            convs_reversed.append(conv_layer)

            out_channels = in_channel_size_
            out_height = in_height
            out_width = in_width

        convs = convs_reversed[::-1]
        self.CNN = Sequential(*convs)
        self._conv_in_shape = (in_channel_size_, out_height, out_width)

        # FEED FORWARD LAYERS -----------------------------------------------------------------
        if ffnn_layer_size is None:
            ffnn_layer_size = [100, 100]
        self._ffnn_layer_size = ffnn_layer_size

        layers = []
        in_size = in_features
        for out_size in ffnn_layer_size:
            layers.append(Linear(in_size, out_size))
            layers.append(ActFunc())
            if dropout is not None:
                layers.append(Dropout(dropout))
            in_size = out_size
        layers.append(Linear(in_size, np.prod(self._conv_in_shape)))

        self.ffnn = Sequential(*layers)

    def forward(self, x):

        x = self.ffnn(x)
        x = x.view((-1,) + self._conv_in_shape)
        x = self.CNN(x)

        return x


class CheckZeros(Module):
    def forward(self, x):
        pass


if __name__ == "__main__":
    from src.data import SkinCancerDataset

    data = SkinCancerDataset(image_size=(225, 300))
    image_size = data.X.shape[1:]

    net = Encoder(
        image_size,
        out_features=4,
        kernel_size=[3, 2],
        padding_size=[2, 1],
        out_channel_size=[7, 3],
        stride=[1, 1],
        activation_function=None,
        ffnn_layer_size=None,
        dropout=None,
        dropout2d=None,
        maxpool=None,
    )

    hej = Decoder(
        in_features=4,
        reshape_features=net.last_im_size,
        out_features=image_size,
        kernel_size=[3, 2],
        padding_size=[2, 1],
        out_channel_size=[7, 3],
        stride=[1, 1],
        activation_function=None,
        ffnn_layer_size=None,
        dropout=None,
        dropout2d=None,
    )

    tmp = hej(net(data.X[0:10, :, :, :]))
