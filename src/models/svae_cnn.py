from src.models.cnn_encoder_decoder import Decoder, Encoder
from src.models.svae import SphericalVAE
import torch
from torch.distributions import Distribution, Normal
from torch.distributions.kl import kl_divergence
from torch.nn import Module
from torch.tensor import Tensor
from src.distributions import VonMisesFisher, SphereUniform
from torch.utils.data import random_split


class SphericalConvolutionalVAE(SphericalVAE):

    def __init__(
        self, image_size, latent_dim
    ):

        super().__init__(feature_dim = image_size[0], latent_dim = latent_dim)

        self.image_size = image_size
        self.latent_dim = latent_dim    

        self.encoder = Encoder(image_size, out_features = latent_dim + 1,  kernel_size = [3, 2], padding_size = [2, 1],
                 out_channel_size = [7, 3], stride = [1,1],
                 activation_function = None, ffnn_layer_size = None,
                 dropout = None, dropout2d = None, maxpool = None)
        print(self.encoder.last_im_size)
    
        self.decoder = Decoder(in_features = latent_dim + 1,
                 reshape_features = self.encoder.last_im_size,
                 out_features = image_size, kernel_size = [3, 2], padding_size =  [2, 1],
                 out_channel_size = [7, 3], stride = [1,1],
                 activation_function = None, ffnn_layer_size = None,
                 dropout = None, dropout2d = None)

        self.to(torch.double)
       
if __name__ == "__main__":
    tmp = SphericalConvolutionalVAE(image_size = [3, 225, 300], latent_dim = 3)
    print("fuckhoved")

