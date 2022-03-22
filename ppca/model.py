'''
Just a wrapping for a Pytorch PPCA
'''
import torch
import numpy as np
from torch import nn
from torch.distributions import Categorical, MixtureSameFamily, Normal, Independent, MultivariateNormal


class trained_PPCA(nn.Module):
    def __init__(self, n_features: int, latent_dim: int, W = None, log_var = None, training_mean = None, device='cpu'):
        super(trained_PPCA, self).__init__()

        self.n_features = n_features
        self.latent_dim = latent_dim
        self.training_mean = training_mean.view(-1,self.n_features)

        self.W = nn.Parameter(W, requires_grad=True)
        self.log_var = nn.Parameter(log_var, requires_grad=True)

        self.device = device

        # here I can also store the other ingredient I

    def compute_log_likelihood(self, example, per_sample_likelihood = True):

        # to get the gradient I should compute all the quantities
        # I need every time
        # Compute C --> remember to take the exp of log_var because we need the var
        id = torch.eye(self.n_features).to(self.device)
        C = self.W @ self.W.T + self.log_var.exp() * id

        # for security check I reshape the example and mu
        example = example.view(-1, self.n_features)

        _, logdetC = torch.slogdet(C)
        precision = torch.inverse(C)
        res = example - self.training_mean

        if per_sample_likelihood:
            # I have to follow the sklearn computation
            # S = (1./example.shape[0]) *(res.T @ res)
            log_like = -0.5 * (self.n_features * np.log(2.0 * np.pi) + logdetC + (res @ precision @ res.T))
            log_like = torch.diag(log_like)
        else:
            log_like = -0.5 * (self.n_features * np.log(2.0 * np.pi) + logdetC + (res @ precision @ res.T))
            log_like = torch.diag(log_like).mean()

        return log_like


class trained_PPCA2(nn.Module):
    def __init__(self, n_features: int, latent_dim: int, W = None, log_var = None, training_mean = None, device='cpu'):
        super(trained_PPCA2, self).__init__()

        self.n_features = n_features
        self.latent_dim = latent_dim
        self.training_mean = nn.Parameter(training_mean, requires_grad=True)

        self.W = nn.Parameter(W, requires_grad=True)
        self.log_var = nn.Parameter(log_var, requires_grad=True)

        self.device = device

        # here I can also store the other ingredient I

    def compute_log_likelihood(self, example):

        # to compute the log-like I can just create a multivariate distribution and compute the
        # log probability by using the log_prob function and take the derivative directly
        id = torch.eye(self.n_features).to(self.device)
        pca_dist = MultivariateNormal(self.training_mean, self.W @ self.W.T + (self.log_var).exp() * id)

        return pca_dist.log_prob(example)



