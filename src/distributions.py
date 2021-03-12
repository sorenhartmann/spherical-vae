from torch.distributions import Distribution, Beta, Normal, constraints

import torch
from torch import Tensor

# From Deep Learning course week 7
class ReparameterizedDiagonalGaussian(Distribution):
    """
    A distribution `N(y | mu, sigma I)` compatible with the reparameterization trick given `epsilon ~ N(0, 1)`.
    """

    def __init__(self, mu: Tensor, log_sigma: Tensor):
        assert (
            mu.shape == log_sigma.shape
        ), f"Tensors `mu` : {mu.shape} and ` log_sigma` : {log_sigma.shape} must be of the same shape"
        self.mu = mu
        self.sigma = log_sigma.exp()

    def sample_epsilon(self) -> Tensor:
        """`\eps ~ N(0, I)`"""
        return torch.empty_like(self.mu).normal_()

    def sample(self) -> Tensor:
        """sample `z ~ N(z | mu, sigma)` (without gradients)"""
        with torch.no_grad():
            return self.rsample()

    def rsample(self) -> Tensor:
        """sample `z ~ N(z | mu, sigma)` (with the reparameterization trick) """
        return self.mu + self.sigma * self.sample_epsilon()

    def log_prob(self, z: Tensor) -> Tensor:
        """return the log probability: log `p(z)`"""
        return torch.distributions.normal.Normal(self.mu, self.sigma).log_prob(z)


class SphereUniform(Distribution):

    arg_constraints = {
        "dim" : torch.distributions.constraints.positive_integer,
    }
    has_rsample = False

    def __init__(self, dim: Tensor, validate_args=None):

        self.dim = torch.tensor(dim)
        self._normal = Normal(0, 1)

        super().__init__(
            torch.Size([dim]),
            validate_args=validate_args,
        )

    def sample(self, shape=torch.Size()):

        norm_sample = self._normal.sample(shape + torch.Size([self.dim+1]) )

        return norm_sample / norm_sample.norm(dim=-1, keepdim=True)


class VonMisesFisher(Distribution):

    arg_constraints = {
        "mu" : constraints.real,
        "kappa" : constraints.positive,
    }

    def __init__(self, mu: Tensor, kappa: Tensor, validate_args=None):
        
        self.mu = mu
        self.kappa = kappa
        self.m = mu.shape[-1]

        #TODO: Fix batch/event size
        self.beta = Beta((self.m-1)/2, (self.m-1)/2)

        super().__init__(
            self.mu.size,
            validate_args=validate_args,
        )
    
    def sample(self) -> Tensor:
        """sample `z ~ N(z | mu, sigma)` (without gradients)"""
        with torch.no_grad():
            return self.rsample()

    def rsample(self) -> Tensor:
        """sample `z ~ N(z | mu, sigma)` (with the reparameterization trick) """

        k = self.k
        m = self.m
        beta_sample = self.beta.rsample

        b = ( -2 * k + torch.sqrt( 4 * k**2 + (m - 1)**2 )) / (m - 1) 
        a = ( (m - 1) + 2 * k + ( 4 * k**2 + (m - 1)**2 ).sqrt() ) / 4
        d = (4 * a * b) / (1 + b) - (m - 1) * torch.log(m - 1)

        while True:

            epsilon = beta_sample()
            w = (1 - (1 + b) * epsilon) / (1 - (1 - b) * epsilon)
            t = 2 * a * b / (1 - (1 - b) * epsilon)

            u = torch.rand(1)

            if float((m - 1) * torch.log(t) - t + d) >= float(torch.log(u)):
                print(
                    float((m - 1) * torch.log(t) - t + d),
                    float(torch.log(u)),
                )
                break

        if m == 2:

            print("Don't sample")

        elif m == 3:

            print("Sample from circle") 
            #TODO: Hør hvorfor man man ikke behøver at sample

        elif m > 3:

            print("Sample uniformly from subsphere")


        return epsilon

