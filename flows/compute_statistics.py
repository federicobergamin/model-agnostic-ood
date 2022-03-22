'''
File used to compute the statistics described in the paper
"Model-agnostic out-of-distribution detection using combined statistical tests"
and to combine them using Fisher's method.

In this file, we are using a Glow implementation from https://github.com/PolinaKirichenko/flows_ood
'''

import argparse
import os
import logging
import sys
sys.path.append("/home/fedbe/flows-ood/")

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torchvision import datasets, transforms, utils

from scipy.spatial.distance import cdist
from torch.nn import functional as F
import seaborn as sns

from flow_ssl import *
# import utils
from experiments.train_flows import utils
from flow_ssl.realnvp import RealNVP
from flow_ssl import FlowLoss
from flow_ssl.glow import Glow
from tqdm import tqdm
from torch import distributions
import torch.nn as nn
import math
from data import make_sup_data_loaders, make_sup_data_loaders2

from tqdm import tqdm
from flow_ssl.data import make_sup_data_loaders
from flow_ssl.invertible.downsample import SqueezeLayer
from statsmodels.distributions.empirical_distribution import ECDF
import pickle as pkl

def to_vector(tensors):
    return torch.cat([t.flatten() for t in tensors])

def get_gradient_vector(parameters):
    """Get single vector of model parameter gradients (where not None)"""
    gradients = [p.grad for p in parameters if p.grad is not None]
    gradient = to_vector(gradients)
    gradient = [g.flatten() for g in gradients]
    gradient = torch.cat(gradient)
    return gradient


def get_parameter_vector(parameters):
    """Get single vector of model parameters"""
    parameters = [p for p in parameters]
    parameter = to_vector(parameters)
    parameter = [g.flatten() for g in parameters]
    parameter = torch.cat(parameter)
    return parameter

parser = argparse.ArgumentParser(description='RealNVP')
parser.add_argument('--dataset', type=str, default="CIFAR10", metavar='DATA',
                help='Dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default='/scratch/fedbe/data/', metavar='PATH',
                help='path to datasets location (default: None)')
parser.add_argument('--optimizer', type=str, default='adam',
                help='optimizer used to trained the model')

args = parser.parse_args()
args.flow = 'Glow'

logger_name = f"statistics_on_{args.flow}_trained_on_{args.dataset}_{args.optimizer}"

LOGGER = logging.Logger(name=logger_name)

LOGGER.info("Args information")
LOGGER.info(f"Model we are using: {args.flow}")
LOGGER.info(f"Dataset: {args.dataset}")
LOGGER.info(f"Data path: {args.data_path}")

saving_directory = "estimated_quantities_glow_trained_on_" + args.dataset.lower()  + '_' + args.optimizer +'/'
os.makedirs(saving_directory + 'distributions', exist_ok=True)
os.makedirs(saving_directory + 'results', exist_ok=True)

# I should set the seed
np.random.seed(1)
torch.manual_seed(1)

# transformation that we should apply to the train, test and validation set
transform_train = transforms.Compose([
             transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.ToTensor()
])

data_path = args.data_path

# load the data
assert args.dataset.lower() == 'fashionmnist' or args.dataset.lower() == 'cifar10' or args.dataset.lower() == 'mnist' or args.dataset.lower() == 'svhn', "You can only decide between FashionMNIST and CIFAR10"
trainloader_in, testloader_in, validloader_in, num_classes = make_sup_data_loaders2(
        args.data_path,
        10,
        4,
        transform_train,
        transform_test,
        use_validation=True,
        shuffle_train=True,
        dataset=args.dataset.lower())

# I should also load the OOD dataset in this case
if args.dataset.lower() == 'fashionmnist':
    # I should load MNIST and use the test_transform

    test_set_ood = datasets.MNIST(args.data_path, train=False, download=True,
                                transform=transform_test)
    testloader_ood = torch.utils.data.DataLoader(
        test_set_ood,
        batch_size=10,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    LOGGER.info("We are training on FashionMNIST (IN) and testing on MNIST (OOD)")
elif args.dataset.lower() == 'cifar10':
    # I should load SVHN and use the test_transform
    test_set_ood = datasets.SVHN(args.data_path, split='test', download=True, transform=transform_test)
    testloader_ood = torch.utils.data.DataLoader(
        test_set_ood,
        batch_size=10,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    LOGGER.info("We are training on CIFAR10 (IN) and testing on SVHN (OOD)")
elif args.dataset.lower() == 'mnist':
    # I should load MNIST and use the test_transform

    test_set_ood = datasets.FashionMNIST(args.data_path, train=False, download=True,
                                transform=transform_test)
    testloader_ood = torch.utils.data.DataLoader(
        test_set_ood,
        batch_size=10,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print('so qua')
    LOGGER.info("We are training on MNIST (IN) and testing on FashionMNIST (OOD)")
elif args.dataset.lower() == 'svhn':
    # I should load MNIST and use the test_transform

    test_set_ood = datasets.CIFAR10(args.data_path, train=False, download=True,
                                transform=transform_test)
    testloader_ood = torch.utils.data.DataLoader(
        test_set_ood,
        batch_size=10,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    LOGGER.info("We are training on SVHN (IN) and testing on CIFAR10 (OOD)")
else:
    raise ValueError("You can only decide between FashionMNIST, CIFAR10, SVHN, MNIST")

# let's check that everything is ok
LOGGER.info(f"Train examples {len(trainloader_in.dataset)}")
LOGGER.info(f"Valid examples {len(validloader_in.dataset)}")
LOGGER.info(f"Test (IN) examples {len(testloader_in.dataset)}")
LOGGER.info(f"Test (OOD) examples {len(testloader_ood.dataset)}")

# now we have loaded our model
print('Security check on the dataset we loaded')

# Model parameters we used to train it
if args.dataset.lower() == 'fashionmnist' or args.dataset.lower() == 'mnist':
    num_mid_channels = 200
    num_blocks = 3
    num_scales = 2
    no_multi_scale = False
    num_coupling_layers_per_scale = 16
    st_type = 'highway'
    img_shape = (1, 28, 28)
else:
    num_mid_channels = 400
    num_blocks = 3
    num_scales = 3
    no_multi_scale = False
    num_coupling_layers_per_scale = 8
    st_type = 'highway'
    img_shape = (3, 32, 32)

print('Building {} model...'.format(args.flow))
model_cfg = Glow


device = 'cuda' if torch.cuda.is_available()  else 'cpu'

# I have to load the weights of the trained model
# todo: add model number here, the one working the best
if args.dataset.lower() == 'fashionmnist':
    if args.optimizer.lower() == 'adam':
        print('GLOW FASHION WITH ADAM')
        params_dir = '../models_fashion_1e5/best_model.pt'
        print(params_dir)
    else:
        print('GLOW FASHION WITH RMSPROP')
        params_dir = '../models_fashion_rmsprop/best_model.pt'
        print(params_dir)
elif args.dataset.lower() == 'cifar10':
    if args.optimizer.lower() == 'adam':
        print('GLOW cifar WITH ADAM')
        params_dir = '../models_cifar_adam/best_model.pt' 
    else:
        print('GLOW cifar WITH rmsprop')
        params_dir = '../models/best_model.pt' 
elif args.dataset.lower() == 'svhn':
    if args.optimizer.lower() == 'adam':
        print('GLOW svhn WITH ADAM')
        params_dir = '../models_svhn_adam/best_model.pt'
    else:
        print('GLOW svhn WITH rmsprop')
        params_dir = '../models_svhn/100.pt'
elif args.dataset.lower() == 'mnist':
    if args.optimizer.lower() == 'adam':
        print('GLOW mnist WITH ADAM')
        params_dir = '../model_mnist_adam/best_model.pt'
        print(params_dir)
    else:
        print('GLOW mnist WITH rmsprop')
        params_dir = '../model_mnist_rmsprop/best_model.pt'

net = model_cfg(image_shape=img_shape, mid_channels=num_mid_channels, num_scales=num_scales,
                    num_coupling_layers_per_scale=num_coupling_layers_per_scale, num_layers=num_blocks,
                    multi_scale=not no_multi_scale, st_type=st_type)

# params_path = params_dir + f'{ckpt}.pt'
checkpoint = torch.load(params_dir)
# net.load_state_dict(checkpoint['net'])
net.set_actnorm_init()
LOGGER.info(f"The model we are using was trained until epoch: {checkpoint['epoch']}")
net = nn.DataParallel(net)
net.load_state_dict(checkpoint['net'])
net = net.to(device)
LOGGER.info("Weights loaded")

# we put the flow in evaluation mode
net.eval()

D = np.prod(img_shape)
D = int(D)

prior = distributions.MultivariateNormal(torch.zeros(D).to(device),
                                             torch.eye(D).to(device))

loss_fn = FlowLoss(prior)


# now I guess we can start the computation using our training set
# now I have to compute the things in the training error
fim_diag = 0
grad_mean = 0
log_like_mean = 0
grad_norm_mean = 0
i = 0
net.zero_grad()
print('starting computation of things from training set')
for batch_idx, (input,_) in tqdm(enumerate(trainloader_in), desc='Computation using the training'):
    for idx, img in enumerate(input):
        if args.dataset.lower() == 'fashionmnist' or args.dataset.lower() == 'mnist':
            example = img.reshape(1, 1, 28, 28)
        else:
            example = img.reshape(1, 3, 32, 32)

        example = example.to(device)
        # net.load_state_dict((torch.load(checkpoint['net'])))
        net.eval()
        net.zero_grad()

        # now I have to pass the input through the network
        # and then compute the loss and the respective gradient
        z = net(example)
        sldj = net.module.logdet()
        loss = loss_fn(z, sldj=sldj)

        (-loss).backward()

        # now I can collect the gradient like I was doing on the PixelCNN++ I suppose
        # _gradient = torch.nn.utils.parameters_to_vector(p.grad for p in net.parameters() if p.grad is not None).detach()
        _gradient = get_gradient_vector(net.parameters()).detach()

        log_like_mean = 1 / (i + 1) * (i * log_like_mean + -loss.detach())
        grad_mean = 1 / (i + 1) * (i * grad_mean + _gradient)
        fim_diag = 1 / (i + 1) * (i * fim_diag + (_gradient) ** 2)
        grad_norm_mean = (torch.norm(_gradient) + i * grad_norm_mean) / (i + 1)

        i += 1
        net.zero_grad()

print('Storing estimated quantities')
torch.save(grad_mean, saving_directory + 'grad_mean_training_estm.pt')
torch.save(log_like_mean, saving_directory + 'log_like_mean_training_estm.pt')
torch.save(grad_norm_mean, saving_directory + 'grad_norm_mean_training_estm.pt')
torch.save(fim_diag, saving_directory + 'fim_diag_training_estm.pt')


# at this point I can get the reciprocal of the fim
fim_approx = (fim_diag + 1e-8)
fim_reciprocal = 1/fim_approx

print('Computing typicality on the validation set')
mmd_fisher_valid = []
mmd_typicality_valid = []
mmd_identity_valid = []
score_statistic_valid = []
grad_norm_valid = []
likelihood_valid = []

for batch_idx, (input,_) in tqdm(enumerate(validloader_in), desc='validation computation'):
    # print('We are evaluating the Validation set, we are at {}/{}'.format(batch_idx, valid_tot_batches))
    for idx, img in enumerate(input):
        if args.dataset.lower() == 'fashionmnist' or args.dataset.lower() == 'mnist':
            example = img.reshape(1, 1, 28, 28)
        else:
            example = img.reshape(1, 3, 32, 32)

        example = example.to(device)
        # net.load_state_dict((torch.load(checkpoint['net'])))
        net.eval()
        net.zero_grad()

        # now I have to pass the input through the network
        # and then compute the loss and the respective gradient
        z = net(example)
        sldj = net.module.logdet()
        loss = loss_fn(z, sldj=sldj)

        (-loss).backward()

        # _grad0 = torch.nn.utils.parameters_to_vector(p.grad for p in net.parameters() if p.grad is not None).detach()
        _grad0 = get_gradient_vector(net.parameters()).detach()

        # and compute the mmd fisher for the example
        likelihood_valid.append((-loss.detach()).item())
        grad_norm_valid.append((torch.norm(_grad0)).item())
        mmd_fisher_valid.append(torch.norm(torch.sqrt(fim_reciprocal) * (grad_mean - _grad0)).item())
        # mmd_typicality_valid.append(torch.norm(log_like_mean - likelihood_data.likelihood).item())
        mmd_typicality_valid.append(torch.norm(log_like_mean - (-loss).detach()).item())
        mmd_identity_valid.append(torch.norm(torch.ones_like(fim_reciprocal) * (grad_mean - _grad0)).item())
        score_statistic_valid.append(torch.norm(torch.sqrt(fim_reciprocal) * _grad0).item())

        net.zero_grad()

print('Storing the validation statistics')
torch.save(mmd_fisher_valid, saving_directory + 'distributions/mmd_fisher_valid.pt')
torch.save(mmd_typicality_valid, saving_directory + 'distributions/mmd_typicality_valid.pt')
torch.save(mmd_identity_valid, saving_directory + 'distributions/mmd_identity_valid.pt')
torch.save(score_statistic_valid, saving_directory + 'distributions/score_statistic_valid.pt')
torch.save(likelihood_valid, saving_directory + 'distributions/log_like_valid.pt')
torch.save(grad_norm_valid, saving_directory + 'distributions/grad_norm_valid.pt')

# now I can compute the distribution of these values
cdf_fisher = ECDF(mmd_fisher_valid)
cdf_typicality = ECDF(mmd_typicality_valid)
cdf_identity = ECDF(mmd_identity_valid)
cdf_score = ECDF(score_statistic_valid)

# now we can evaluate the TEST SET (IN)
print('starting computation of the TEST SET IN-DISTRIBUTION')
mmd_fisher_in = []
p_values_fisher_in = []
mmd_typicality_in = []
p_values_typicality_in = []
mmd_identity_in = []
p_values_identity_in = []
score_statistic_in = []
p_values_score_in = []
grad_norm_in = []
log_like_in = []

print('starting evaluation IN-test set')
for batch_idx, (batch,_) in tqdm(enumerate(testloader_in), desc='In-distribution test set'):
    for idx, example in enumerate(batch):
        if args.dataset.lower() == 'fashionmnist' or args.dataset.lower() == 'mnist':
            example = example.reshape(1, 1, 28, 28)
        else:
            example = example.reshape(1, 3, 32, 32)

        example = example.to(device)
        # net.load_state_dict((torch.load(checkpoint['net'])))
        net.eval()
        net.zero_grad()

        # now I have to pass the input through the network
        # and then compute the loss and the respective gradient
        z = net(example)
        sldj = net.module.logdet()
        loss = loss_fn(z, sldj=sldj)

        (-loss).backward()

        # _grad0 = torch.nn.utils.parameters_to_vector(p.grad for p in net.parameters() if p.grad is not None).detach()
        _grad0 = get_gradient_vector(net.parameters()).detach()

        log_like_in.append((-loss).detach().item())
        grad_norm_in.append(torch.norm(_grad0).item())

        _mmd_fisher_score = torch.norm(torch.sqrt(fim_reciprocal) * (grad_mean - _grad0)).item()
        mmd_fisher_in.append(_mmd_fisher_score)
        p_values_fisher_in.append(1 - cdf_fisher(_mmd_fisher_score))

        # _typicality_score = torch.norm(log_like_mean - likelihood_data.likelihood).item()
        _typicality_score = torch.norm(log_like_mean - (-loss).detach()).item()
        mmd_typicality_in.append(_typicality_score)
        p_values_typicality_in.append(1 - cdf_typicality(_typicality_score))

        _mmd_identity_score = torch.norm(torch.ones_like(fim_reciprocal) * (grad_mean - _grad0)).item()
        mmd_identity_in.append(_mmd_identity_score)
        p_values_identity_in.append(1 - cdf_identity(_mmd_identity_score))

        _fisher_score = torch.norm(torch.sqrt(fim_reciprocal) * _grad0).item()
        score_statistic_in.append(_fisher_score)
        p_values_score_in.append(1 - cdf_score(_fisher_score))

        net.zero_grad()

print('starting evaluation on the OOD test set')
mmd_fisher_ood = []
p_values_fisher_ood = []
mmd_typicality_ood = []
p_values_typicality_ood = []
mmd_identity_ood = []
p_values_identity_ood = []
score_statistic_ood = []
p_values_score_ood = []
grad_norm_ood = []
log_like_ood = []
for batch_idx, (batch,_) in tqdm(enumerate(testloader_ood), desc='OOD dataset computation'):
    for idx, example in enumerate(batch):
        if args.dataset.lower() == 'fashionmnist' or args.dataset.lower() == 'mnist':
            example = example.reshape(1, 1, 28, 28)
        else:
            example = example.reshape(1, 3, 32, 32)

        example = example.to(device)
        # net.load_state_dict((torch.load(checkpoint['net'])))
        net.eval()
        net.zero_grad()

        # now I have to pass the input through the network
        # and then compute the loss and the respective gradient
        z = net(example)
        sldj = net.module.logdet()
        loss = loss_fn(z, sldj=sldj)

        (-loss).backward()

        # _grad0 = torch.nn.utils.parameters_to_vector(p.grad for p in net.parameters() if p.grad is not None).detach()
        _grad0 = get_gradient_vector(net.parameters()).detach()

        # now I can compute the typicality test and p-values
        # log_like_ood.append(likelihood_data.likelihood.item())
        log_like_ood.append((-loss).detach().item())
        grad_norm_ood.append(torch.norm(_grad0).item())

        _mmd_fisher_score = torch.norm(torch.sqrt(fim_reciprocal) * (grad_mean - _grad0)).item()
        mmd_fisher_ood.append(_mmd_fisher_score)
        p_values_fisher_ood.append(1 - cdf_fisher(_mmd_fisher_score))

        # _typicality_score = torch.norm(log_like_mean - likelihood_data.likelihood).item()
        _typicality_score = torch.norm(log_like_mean - (-loss).detach().item()).item()
        mmd_typicality_ood.append(_typicality_score)
        p_values_typicality_ood.append(1 - cdf_typicality(_typicality_score))

        _mmd_identity_score = torch.norm(torch.ones_like(fim_reciprocal) * (grad_mean - _grad0)).item()
        mmd_identity_ood.append(_mmd_identity_score)
        p_values_identity_ood.append(1 - cdf_identity(_mmd_identity_score))

        _fisher_score = torch.norm(torch.sqrt(fim_reciprocal) * _grad0).item()
        score_statistic_ood.append(_fisher_score)
        p_values_score_ood.append(1 - cdf_score(_fisher_score))

        net.zero_grad()

# now I have to store everything

with open(saving_directory + 'results/in_dist_score_mmd_fisher.pickle', 'wb') as handle:
    pkl.dump(mmd_fisher_in, handle, protocol=pkl.HIGHEST_PROTOCOL)

with open(saving_directory + 'results/in_dist_mmd_typicality.pickle', 'wb') as handle:
    pkl.dump(mmd_typicality_in, handle, protocol=pkl.HIGHEST_PROTOCOL)

with open(saving_directory + 'results/in_dist_mmd_fisher_identity.pickle', 'wb') as handle:
    pkl.dump(mmd_identity_in, handle, protocol=pkl.HIGHEST_PROTOCOL)

with open(saving_directory + 'results/in_dist_score_statistic.pickle', 'wb') as handle:
    pkl.dump(score_statistic_in, handle, protocol=pkl.HIGHEST_PROTOCOL)

with open(saving_directory + 'results/in_dist_p_values_fisher.pickle', 'wb') as handle:
    pkl.dump(p_values_fisher_in, handle, protocol=pkl.HIGHEST_PROTOCOL)

with open(saving_directory + 'results/in_dist_p_values_typicality.pickle', 'wb') as handle:
    pkl.dump(p_values_typicality_in, handle, protocol=pkl.HIGHEST_PROTOCOL)

with open(saving_directory + 'results/in_dist_p_values_fisher_identity.pickle', 'wb') as handle:
    pkl.dump(p_values_identity_in, handle, protocol=pkl.HIGHEST_PROTOCOL)

with open(saving_directory + 'results/in_dist_p_values_score_stat.pickle', 'wb') as handle:
    pkl.dump(p_values_score_in, handle, protocol=pkl.HIGHEST_PROTOCOL)

with open(saving_directory + 'results/in_dist_grad_norm.pickle', 'wb') as handle:
    pkl.dump(grad_norm_in, handle, protocol=pkl.HIGHEST_PROTOCOL)

with open(saving_directory + 'results/in_dist_log_like.pickle', 'wb') as handle:
    pkl.dump(log_like_in, handle, protocol=pkl.HIGHEST_PROTOCOL)


## -------------------- MNIST test set -------------------
with open(saving_directory + 'results/ood_mmd_fisher.pickle', 'wb') as handle:
    pkl.dump(mmd_fisher_ood, handle, protocol=pkl.HIGHEST_PROTOCOL)

with open(saving_directory + 'results/ood_mmd_typicality.pickle', 'wb') as handle:
    pkl.dump(mmd_typicality_ood, handle, protocol=pkl.HIGHEST_PROTOCOL)

with open(saving_directory + 'results/ood_mmd_fisher_identity.pickle', 'wb') as handle:
    pkl.dump(mmd_identity_ood, handle, protocol=pkl.HIGHEST_PROTOCOL)

with open(saving_directory + 'results/ood_score_statistic.pickle', 'wb') as handle:
    pkl.dump(score_statistic_ood, handle, protocol=pkl.HIGHEST_PROTOCOL)

with open(saving_directory + 'results/ood_p_values_fisher.pickle', 'wb') as handle:
    pkl.dump(p_values_fisher_ood, handle, protocol=pkl.HIGHEST_PROTOCOL)

with open(saving_directory + 'results/ood_p_values_typicality.pickle', 'wb') as handle:
    pkl.dump(p_values_typicality_ood, handle, protocol=pkl.HIGHEST_PROTOCOL)

with open(saving_directory + 'results/ood_p_values_fisher_identity.pickle', 'wb') as handle:
    pkl.dump(p_values_identity_ood, handle, protocol=pkl.HIGHEST_PROTOCOL)

with open(saving_directory + 'results/ood_p_values_score_stat.pickle', 'wb') as handle:
    pkl.dump(p_values_score_ood, handle, protocol=pkl.HIGHEST_PROTOCOL)

with open(saving_directory + 'results/ood_grad_norm.pickle', 'wb') as handle:
    pkl.dump(grad_norm_ood, handle, protocol=pkl.HIGHEST_PROTOCOL)

with open(saving_directory + 'results/ood_log_likelihood.pickle', 'wb') as handle:
    pkl.dump(log_like_ood, handle, protocol=pkl.HIGHEST_PROTOCOL)


# I can also already compute the AUROC scores and printing them
# on the logger

from sklearn.metrics import roc_auc_score
from scipy.stats import chi2

p_values_typicality_in = np.array(p_values_typicality_in)
p_values_score_in = np.array(p_values_score_in)

p_values_typicality_ood = np.array(p_values_typicality_ood)
p_values_score_ood = np.array(p_values_score_ood)

y_ood = [1] * len(p_values_typicality_ood)
y_in = [0] * len(p_values_typicality_in)
y_true = y_ood + y_in
scores = np.concatenate((p_values_typicality_ood,p_values_typicality_in))
print('TYPICALITY AUROC: ', 1-roc_auc_score(y_true, scores))
LOGGER.info(f"TYPICALITY AUROC: {1-roc_auc_score(y_true, scores)}")



scores = np.concatenate((p_values_score_ood,p_values_score_in))
print('SCORE AUROC: ', 1-roc_auc_score(y_true, scores))
LOGGER.info(f"SCORE AUROC: {1-roc_auc_score(y_true, scores)}")


# combined p-values
combined_p_values_in = -2 * (np.log(p_values_typicality_in + 1e-8) + np.log(p_values_score_in+ 1e-8))
combined_p_values_out = -2 * (np.log(p_values_typicality_ood + 1e-8) + np.log(p_values_score_ood + 1e-8))

scores = np.concatenate((combined_p_values_out,combined_p_values_in))
print('FISHER COMBINATION AUROC: ', roc_auc_score(y_true, scores))
LOGGER.info(f"FISHER COMBINATION AUROC: {roc_auc_score(y_true, scores)}")

# maybe I can also try to put the p-values together by using the harmonic stuff
harmonic_mean_in = (2 * p_values_typicality_in * p_values_score_in) / (p_values_typicality_in + p_values_score_in + 1e-8)
harmonic_mean_ood =(2 * p_values_typicality_ood * p_values_score_ood) / (p_values_typicality_ood + p_values_score_ood + 1e-8)

scores = np.concatenate((harmonic_mean_ood,harmonic_mean_in))

print(1-roc_auc_score(y_true, scores))



