'''
Definition of a GMM model. This should contain methods as:
- compute_log_likelihood()

these for now are the two needed methods
'''

import torch
from torch import nn
from torch.distributions import Categorical, MixtureSameFamily, Normal, Independent

class GMM(nn.Module):
    def __init__(self, n_components: int, input_dim: int, mixing_logits = None, means = None, log_var=None):
        super(GMM, self).__init__()
        self.n_components = n_components
        self.input_dim = input_dim

        if mixing_logits is None and means is None and log_var is None:
            # todo: if I want to train this method using gradient descent
            #  a thing that I need is the initalization of the means using k-means

            # if everything is None we should initialize them
            # we are learning both the mixing prob, the mean and the log_var of the different components
            self.mixing_logits = nn.Parameter(torch.ones(self.n_components) / self.n_components, requires_grad=True)
            self.means = nn.Parameter(torch.randn((self.n_components, self.input_dim)), requires_grad=True)
            self.log_var = nn.Parameter(torch.randn((self.n_components, self.input_dim)), requires_grad=True)
        else:
            self.mixing_logits = nn.Parameter(mixing_logits, requires_grad=True)
            self.means = nn.Parameter(means, requires_grad=True)
            self.log_var = nn.Parameter(log_var, requires_grad=True)


    def compute_log_likelihood(self, x):
        # I have to create the gmm distribution using the current mixing logits, means and log_vars
        categorical_dist = Categorical(logits=self.mixing_logits)
        components_dist = Independent(Normal(
             self.means, (0.5 * self.log_var).exp()), 1)

        gmm = MixtureSameFamily(categorical_dist, components_dist)

        # now I have a GMM ready, I can compute the log_prob
        log_prob = gmm.log_prob(x)

        return log_prob


class trained_GMM(nn.Module):
    def __init__(self, n_components: int, mixing_logits = None, means = None, log_var=None):
        super(trained_GMM, self).__init__()
        self.n_components = n_components

        self.mixing_logits = nn.Parameter(mixing_logits, requires_grad=True)
        self.means = nn.Parameter(means, requires_grad=True)
        self.log_var = nn.Parameter(log_var, requires_grad=True)

    def compute_log_likelihood(self, x):
        # I can create the final/trained model
        categorical_dist = Categorical(logits=self.mixing_logits)
        components_dist = Independent(Normal(
            self.means, (0.5 * self.log_var).exp()), 1)

        gmm = MixtureSameFamily(categorical_dist, components_dist)

        return gmm.log_prob(x)


