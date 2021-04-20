import torch
from torch.nn import Module, Sequential, Linear, ReLU,Conv2d, Dropout2d, BatchNorm2d, MaxPool2d

def compute_conv_dim(dim_size, kernel_size, padding, stride):
    return int((dim_size - kernel_size + 2 * padding) / stride + 1)

def compute_conv_transpose_dim(dim_size, kernel_size, padding, dialation, stride):
    return int((dim_size-1)*stride-2*padding+dialation*(kernel_size-1)+1)

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
            convs.append(BatchNorm2d(parameters[1])) 
           
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

    def __init__(self, in_features, reshape_features, out_features, kernel_size, stride):

        super().__init__()

        #channels, height, width = image_size
        self.reshape_features = reshape_features

        # Parameters for CNN part 
        conv_1_out_channels =  10 
        conv_1_kernel_size = 7 
        conv_1_stride = 1 
        conv_1_pad    = 4

        conv1_height = compute_conv_dim(self.reshape_features[1], conv_1_kernel_size, conv_1_pad, conv_1_stride)
        conv1_width = compute_conv_dim(self.reshape_features[2], conv_1_kernel_size, conv_1_pad, conv_1_stride)

        conv_2_out_channels =  3
        conv_2_kernel_size = 11 
        conv_2_stride = 4 
        conv_2_pad    = 3 

        conv2_height = compute_conv_dim(conv1_height, conv_2_kernel_size, conv_2_pad, conv_2_stride)
        #print(conv2_height)
        conv2_width = compute_conv_dim(conv1_width, conv_2_kernel_size, conv_2_pad, conv_2_stride)
        #print(conv2_width)

        self.ffnn = Sequential(
            Linear(in_features=in_features, out_features=100),
            ReLU(),
            Linear(in_features=100, out_features=100),
            ReLU(),
            Linear(in_features=100, out_features=np.prod(self.reshape_features))
        ) 

        self.CNN = Sequential(
            #--------------------------
            # First convolutional layer 
            #Conv2d(
            #    in_channels=self.reshape_features[0],
            #    out_channels=conv_1_out_channels,
            #    kernel_size=conv_1_kernel_size,
            #    stride=conv_1_stride,
            #    padding=conv_1_pad),
            # Batch normalization
            #BatchNorm2d(conv_1_out_channels),
            #Dropout2d(p = 0.5),
            #Activation function
            #ReLU(),

            # --------------------------
            # Second convolutional layer 
            #Conv2d(
            #    in_channels=conv_1_out_channels,
            #    out_channels=conv_2_out_channels,
            #    kernel_size=conv_2_kernel_size,
            #    stride=conv_2_stride,
            #    padding=conv_2_pad),
            # Batch Normalization 
            #BatchNorm2d(conv_2_out_channels),
            # Activation function 
            #ReLU(),

            # --------------------------
            # First transposed convolutional layer 
            ConvTranspose2d(in_channels=conv_2_out_channels, 
                out_channels = conv_2_out_channels,
                kernel_size=(2, 4),
                stride = 2,
                padding = 3),
            # Batch Normalization 
            #BatchNorm2d(conv_2_out_channels),
            # Activation function 
            #ReLU(),
        
            # --------------------------
            # Second transposed convolutional layer 
            ConvTranspose2d(in_channels= conv_2_out_channels, 
                out_channels = 3,
                kernel_size=(2, 4),
                stride = 2,
                padding = 3),
            
            # --------------------------
            # Third transposed convolutional layer
            ConvTranspose2d(in_channels= 3, 
                out_channels = 3,
                kernel_size=(2, 4),
                stride = 2,
                padding = 3)
        )


    def forward(self, x):
        
        x = self.ffnn(x)
        new_dim = (x.shape[0],) + self.reshape_features
        print(new_dim) 
        x = self.CNN(x.view(new_dim))
        

        print(x.shape)
        x = self.CNN(x.view(x.shape[0], conv_2_out_channels, maxPool_height, maxPool_width))
        print(x.shape)
        return x


if __name__ == "__main__":
    from src.data import SkinCancerDataset
    data = SkinCancerDataset(image_size=(225, 300))
    image_size = data.X.shape[1:]

    net = Encoder(image_size, out_features = 4,  kernel_size = [3, 2], padding_size = [2, 1],
                 out_channel_size = [7, 3], stride = [1,1],
                 activation_function = None, ffnn_layer_size = None,
                 dropout = None, dropout2d = None, maxpool = None)
    #hej = Decoder(4, image_size)

    #hej(net(data.X[0:10,:,:,:]))