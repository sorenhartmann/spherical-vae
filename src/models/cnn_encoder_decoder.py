import torch
import numpy as np 
from torch.nn import Module, Sequential, Linear, Sigmoid, ReLU,Conv2d, Dropout2d, BatchNorm2d, MaxPool2d, ConvTranspose2d

def compute_conv_dim(dim_size, kernel_size, padding, stride):
    return int((dim_size - kernel_size + 2 * padding) / stride + 1)

def compute_conv_transpose_dim(dim_size, kernel_size, padding, dialation, stride):
    return int((dim_size-1)*stride-2*padding+dialation*(kernel_size-1)+1)

def compute_last_kernel_dim(out_dim_size, dim_size):
    # For fixed stride=1, padding = 0, dialation=1 
    return int(-dim_size + out_dim_size + 1)


class Encoder(Module):
    # kernel_size: list of kernel sizes
    # padding_size: list of padding sizes
    # stride: list of strides
    def __init__(self, image_size,  out_features, kernel_size = None, padding_size = None,
                 out_channel_size = None, stride = None,
                 activation_function = None, ffnn_layer_size = None,
                 dropout = None, dropout2d = None, maxpool = None):

        super().__init__()
        
        self.dropout = dropout
        self.dropout2d = dropout2d

        if activation_function is None:
            activation_function = "ReLU"
        self._activation_function = activation_function

        ActFunc = getattr(torch.nn.modules.activation, activation_function)

        if maxpool is None: 
            maxpool = [2, 2]
         
        # CONVOLUTIONAL LAYERS ------------------------------------------------------------------
        convs = []
        channels, height, width = image_size
        for parameters in list(zip(out_channel_size, kernel_size, stride, padding_size)):
            convs.append(Conv2d(
                in_channels=channels,
                out_channels=parameters[0],
                kernel_size=parameters[1],
                stride=parameters[2],
                padding=parameters[3]
            ))
            convs.append(BatchNorm2d(parameters[0])) 
           
            if dropout is not None:
                convs.append(Dropout2d(dropout2d))
            
            convs.append(ActFunc())
            
            height = compute_conv_dim(height, parameters[1], parameters[3], parameters[2])
            width = compute_conv_dim(width, parameters[1], parameters[3], parameters[2])
            channels = parameters[0]

        convs.append(MaxPool2d(
                kernel_size = maxpool[0],
                stride = maxpool[1]
            ))
        height = compute_conv_dim(height, maxpool[0], 0, maxpool[1])
        width = compute_conv_dim(width, maxpool[0], 0, maxpool[1])

        self.CNN = Sequential(*convs)

        self.linear_in_features = channels*height*width   
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

    def __init__(self, in_features, reshape_features, out_features, kernel_size = None,
                 padding_size = None, out_channel_size = None, stride = None,
                 activation_function = None, ffnn_layer_size = None,
                 dropout = None, dropout2d = None):

        super().__init__()

        self.reshape_features = reshape_features

        self.dropout = dropout
        self.dropout2d = dropout2d

        if activation_function is None:
            activation_function = "ReLU"
        self._activation_function = activation_function

        ActFunc = getattr(torch.nn.modules.activation, activation_function)

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
        layers.append(Linear(in_size, np.prod(reshape_features)))

        self.ffnn = Sequential(*layers)

        # TRANSPOSED CONVOLUTIONAL LAYERS ------------------------------------------------------
        convs = []
        channels, height, width = self.reshape_features
        for parameters in list(zip(out_channel_size, kernel_size, stride, padding_size)):
            convs.append(ConvTranspose2d(
                in_channels=channels,
                out_channels=parameters[0],
                kernel_size=parameters[1],
                stride=parameters[2],
                padding=parameters[3]
            ))

            convs.append(BatchNorm2d(parameters[0])) 
           
            if dropout is not None:
                convs.append(Dropout2d(dropout2d))
            
            convs.append(ActFunc())

            height = compute_conv_transpose_dim(height, parameters[1], parameters[3], 1, parameters[2])
            width = compute_conv_transpose_dim(width, parameters[1], parameters[3], 1, parameters[2])
            channels = parameters[0]
            print(f"channels:{channels}, height:{height}, width:{width}")

        # "Manuel" layer to ensure output image size, stride padding and dialation are default sizes
        k1 = compute_last_kernel_dim(out_dim_size = out_features[1], dim_size = height)
        k2 = compute_last_kernel_dim(out_dim_size = out_features[2], dim_size = width)
        kernel_size = (k1,k2)

        convs.append(ConvTranspose2d(
            in_channels=channels,
            out_channels=out_features[0],
            kernel_size=kernel_size,
        ))
        convs.append(BatchNorm2d(out_features[0]))
        convs.append(ActFunc())

        self.CNN = Sequential(*convs)

    def forward(self, x):
        x = self.ffnn(x)
        new_dim = (x.shape[0],) + self.reshape_features
        x = x.view(new_dim)
        x = self.CNN(x)
        return x

class CheckZeros(Module):
    def forward(self, x):
        pass

if __name__ == "__main__":
    from src.data import SkinCancerDataset
    data = SkinCancerDataset(image_size=(225, 300))
    image_size = data.X.shape[1:]

    net = Encoder(image_size, out_features = 4,  kernel_size = [3, 2], padding_size = [2, 1],
                 out_channel_size = [7, 3], stride = [1,1],
                 activation_function = None, ffnn_layer_size = None,
                 dropout = None, dropout2d = None, maxpool = None)
    
    hej = Decoder(in_features = 4, reshape_features = net.last_im_size,
                 out_features = image_size, kernel_size = [3, 2], padding_size =  [2, 1],
                 out_channel_size = [7, 3], stride = [1,1],
                 activation_function = None, ffnn_layer_size = None,
                 dropout = None, dropout2d = None)

    tmp = hej(net(data.X[0:10,:,:,:]))
