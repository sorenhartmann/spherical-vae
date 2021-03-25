import torch
from torch import Tensor
from torch.distributions import Beta, Distribution, Normal, constraints
import math

from scipy.special import iv


class SphereUniform(Distribution):

    arg_constraints = {}

    has_rsample = False

    def __init__(self, dim: int, batch_shape=torch.Size(), validate_args=None):

        self.dim = dim
        n = dim + 1

        log_surface_area = math.log(2) + n / 2 * math.log(math.pi) - math.lgamma(n / 2)
        self._log_prob = (
            -log_surface_area
            if type(log_surface_area) == torch.Tensor
            else -torch.tensor(log_surface_area)
        )

        super().__init__(
            event_shape=torch.Size([dim + 1]),
            batch_shape=batch_shape,
            validate_args=validate_args,
        )

    def sample(self, sample_shape=torch.Size()):

        sample_shape = (
            sample_shape
            if type(sample_shape) == torch.Size
            else torch.Size(sample_shape)
        )

        norm_sample = torch.zeros(
            sample_shape + self._batch_shape + self._event_shape
        )
        norm_sample.normal_()

        return norm_sample / norm_sample.norm(dim=-1, keepdim=True)

    def log_prob(self, value):

        return torch.tile(self._log_prob, value.shape)


class VonMisesFisher(Distribution):

    arg_constraints = {
        "mu": constraints.real,
        "k": constraints.positive,
    }

    def __init__(self, mu: Tensor, k: Tensor, validate_args=None):

        batch_shape = mu.shape[:-1]
        event_shape = mu.shape[-1:]

        self.mu = mu
        self.k = k
        m = torch.tensor(mu.shape[-1])
        self.m = m

        self._log_2_pi = torch.log(torch.tensor(2 * math.pi))

        self.beta_dist = torch.distributions.Beta((m - 1) / 2, (m - 1) / 2)
        self.uniform_subsphere_dist = SphereUniform(m - 2, batch_shape=batch_shape)

        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def sample(self, sample_shape=torch.Size()) -> Tensor:
        with torch.no_grad():
            return self.rsample()

    def rsample(self, sample_shape=torch.Size()) -> Tensor:

        batch_shape = self._batch_shape

        mu = self.mu
        k = self.k
        m = self.m

        b = (-2 * k + torch.sqrt(4 * k ** 2 + (m - 1) ** 2)) / (m - 1)
        a = ((m - 1) + 2 * k + torch.sqrt(4 * k ** 2 + (m - 1) ** 2)) / 4
        d = (4 * a * b) / (1 + b) - (m - 1) * torch.log(m - 1)

        e = torch.tensor([1] + [0] * (m - 1))
        u_prime = e - mu
        u = u_prime / u_prime.norm(dim=-1, keepdim=True)
        U = torch.eye(m) - 2 * u.view(-1, m, 1) @ u.view(-1, 1, m)

        beta_dist = self.beta_dist
        uniform_subsphere_dist = self.uniform_subsphere_dist

        done = torch.zeros(sample_shape + batch_shape, dtype=bool)
        w = torch.zeros(sample_shape + batch_shape, dtype=torch.float)

        while True:

            mask = ~done
            n_left = mask.sum()

            if n_left == 0:
                break

            # TODO: probably can't train until kl term is there, k is too large 
            a_ = torch.masked_select(a, mask)
            b_ = torch.masked_select(b, mask)
            d_ = torch.masked_select(d, mask)

            epsilon = beta_dist.rsample((n_left,))

            w_proposal = (1 - (1 + b_) * epsilon) / (1 - (1 - b_) * epsilon)
            t = 2 * a_ * b_ / (1 - (1 - b_) * epsilon)

            u = torch.rand(n_left)
            accepted = (m - 1) * torch.log(t) - t + d_ >= torch.log(u)

            # Fix for mask inplace stuff with masked_select
            mask_clone = mask.clone()

            mask_clone[mask] = accepted 
            w[mask_clone] = w_proposal[accepted]
            done[mask_clone] = True


        w = w.view(sample_shape + batch_shape + (1,))

        # Sample from subsphere
        v = uniform_subsphere_dist.sample(sample_shape)

        z_prime = torch.cat([w, v * torch.sqrt(1 - w ** 2)], -1)

        z = (
            U.view(torch.Size(batch_shape + (m, m)))
            @ z_prime.view(sample_shape + batch_shape + (m, 1))
        ).squeeze()

        return z

    def log_prob(self, value):

        m = self.m
        k = self.k

        log_C = (
            (m / 2 - 1) * torch.log(k)
            - (m / 2) * self._log_2_pi
            - torch.log(iv(m / 2 - 1, k))
        )

        return log_C + k * torch.sum(self.mu * value, -1)


def _vmf_uniform_kl(k, m, include_offset=False):
    pass



def _d_vmf_uniform_kl(k):
    pass
