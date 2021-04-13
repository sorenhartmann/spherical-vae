from torch.nn import Module, Sequential, Linear, ReLU,Conv2d, Dropout2d, BatchNorm2d, MaxPool2d

def compute_conv_dim(dim_size, kernel_size, padding, stride):
    return int((dim_size - kernel_size + 2 * padding) / stride + 1)

class Encoder(Module):

    def __init__(self, image_size, out_features):

        super().__init__()
        channels, height, width = image_size

        # Parameters for CNN part 
        conv_1_out_channels =  13 
        conv_1_kernel_size = 7 
        conv_1_stride = 1 
        conv_1_pad    = 4

        conv1_height = compute_conv_dim(height, conv_1_kernel_size, conv_1_pad, conv_1_stride)
        conv1_width = compute_conv_dim(width, conv_1_kernel_size, conv_1_pad, conv_1_stride)

        conv_2_out_channels =  3 
        conv_2_kernel_size = 11 
        conv_2_stride = 4 
        conv_2_pad    = 3 

        conv2_height = compute_conv_dim(conv1_height, conv_2_kernel_size, conv_2_pad, conv_2_stride)
        conv2_width = compute_conv_dim(conv1_width, conv_2_kernel_size, conv_2_pad, conv_2_stride)

        # Parameters for maxpool 
        maxPool_kernel = 2
        maxPool_stride = 2

        maxPool_height = compute_conv_dim(conv2_height, maxPool_kernel, 0, maxPool_stride)
        maxPool_width = compute_conv_dim(conv2_width, maxPool_kernel, 0, maxPool_stride)

        self.linear_in_features = conv_2_out_channels*maxPool_height*maxPool_width       

        self.CNN = Sequential(
            #--------------------------
            # First convolutional layer 
            Conv2d(
                in_channels=channels,
                out_channels=conv_1_out_channels,
                kernel_size=conv_1_kernel_size,
                stride=conv_1_stride,
                padding=conv_1_pad),
            # Batch normalization
            BatchNorm2d(conv_1_out_channels),
            Dropout2d(p = 0.5),
            #Activation function
            ReLU(),

            # --------------------------
            # Second convolutional layer 
            Conv2d(
                in_channels=conv_1_out_channels,
                out_channels=conv_2_out_channels,
                kernel_size=conv_2_kernel_size,
                stride=conv_2_stride,
                padding=conv_2_pad),
            # Batch Normalization 
            BatchNorm2d(conv_2_out_channels),
            Dropout2d(p = 0.5),
            # Activation function 
            ReLU(),

            #----------------------------
            # Max Pooling 
            MaxPool2d(
                kernel_size = maxPool_kernel,
                stride = maxPool_stride
            )
        )

        self.ffnn = Sequential(
            Linear(in_features=self.linear_in_features, out_features=100), 
            ReLU(),
            Linear(in_features=100, out_features=100),
            ReLU(),
            Linear(in_features=100, out_features=out_features),
        )

    def forward(self, x):

        x = self.CNN(x)
        x = self.ffnn(x.view(-1, self.linear_in_features))

        return x


class Decoder(Module):

    def __init__(self, in_features, out_features):

        super().__init__()

        channels, height, width = image_size

        # Parameters for CNN part 
        conv_1_out_channels =  13 
        conv_1_kernel_size = 7 
        conv_1_stride = 1 
        conv_1_pad    = 4

        conv1_height = compute_conv_dim(height, conv_1_kernel_size, conv_1_pad, conv_1_stride)
        conv1_width = compute_conv_dim(width, conv_1_kernel_size, conv_1_pad, conv_1_stride)

        conv_2_out_channels =  3 
        conv_2_kernel_size = 11 
        conv_2_stride = 4 
        conv_2_pad    = 3 

        conv2_height = compute_conv_dim(conv1_height, conv_2_kernel_size, conv_2_pad, conv_2_stride)
        conv2_width = compute_conv_dim(conv1_width, conv_2_kernel_size, conv_2_pad, conv_2_stride)

        # Parameters for maxpool 
        maxPool_kernel = 2
        maxPool_stride = 2

        maxPool_height = compute_conv_dim(conv2_height, maxPool_kernel, 0, maxPool_stride)
        maxPool_width = compute_conv_dim(conv2_width, maxPool_kernel, 0, maxPool_stride)

        self.linear_out_features = conv_2_out_channels*maxPool_height*maxPool_width  

        self.ffnn = Sequential(
            Linear(in_features=in_features, out_features=100),
            ReLU(),
            Linear(in_features=100, out_features=100),
            ReLU(),
            Linear(in_features=100, out_features=self.linear_out_features)
        ) 

        self.CNN = Sequential(
            #--------------------------
            # First convolutional layer 
            Conv2d(
                in_channels=channels,
                out_channels=conv_1_out_channels,
                kernel_size=conv_1_kernel_size,
                stride=conv_1_stride,
                padding=conv_1_pad),
            # Batch normalization
            BatchNorm2d(conv_1_out_channels),
            Dropout2d(p = 0.5),
            #Activation function
            ReLU(),

            # --------------------------
        )

    def forward(self, x):
        
        x = self.ffnn(x)
        print(x.shape)
        x = self.CNN(x.view(x.shape[0], conv_2_out_channels, maxPool_height, maxPool_width))

        return x


if __name__ == "__main__":
    from src.data import SkinCancerDataset
    data = SkinCancerDataset()
    image_size = data.X.shape[1:]

    net = Encoder(image_size, out_features = 4)
    hej = Decoder(4, image_size)

    hej(net(data.X[0:10,:,:,:]))