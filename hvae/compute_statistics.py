'''
File used to compute the statistics described in the paper
"Model-agnostic out-of-distribution detection using combined statistical tests"
and to combine them using Fisher's method.

In this file, we are using a HVAE implementation from https://github.com/JakobHavtorn/hvae-oodd

'''

import argparse
import os
import logging

from collections import defaultdict
from typing import *

from tqdm import tqdm

import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics

import oodd.datasets
import oodd.evaluators
import oodd.models
import oodd.losses
import oodd.utils

from datautils import *
from statsmodels.distributions.empirical_distribution import ECDF
import pickle as pkl

LOGGER = logging.getLogger()


parser = argparse.ArgumentParser()
parser.add_argument("--model_dataset", type=str, default="FashionMNIST", help="")
parser.add_argument("--dequantized", type=int, default=1, help="1- dequantized, 0- binarized")
parser.add_argument("--n_samples", type=int, default=1, help="Number of sampoles we should compute the VAE bound")
parser.add_argument("--test_statistic", type=str, choices=["score", "mmd_fisher", "mmd_typicality"], help="which test statistic to use")
parser.add_argument("--fisher_approx_source", type=str, choices=["data", "model", "None"], help="whether to estimate model fisher matrix using training data or model samples. If None, use identity matrix.")
parser.add_argument("--fisher_approx_n_samples", type=int, default=None, help="number of samples for fisher matrix approxition. Defaults to lowest number of samples in test sets.")
parser.add_argument("--eps", type=float, default=1e-8, help="value eps in Fisher matrix approximation")
parser.add_argument("--xi", type=float, default=1, help="Value of xi in Fisher matrix approximation")
parser.add_argument("--n_posterior_samples", type=int, default=1, help="")
parser.add_argument("--seed", type=int, default=1, help="Seed for reproducibily")

parser.add_argument(
    "--generative_parameters_only",
    action="store_true",
    default=False,
    help="If true, evaluate gradient only on generative model parameters",
)
parser.add_argument(
    "--inference_parameters_only",
    action="store_true",
    default=False,
    help="If true, evaluate gradient only on inference model parameters",
)
parser.add_argument("--n_evaluation_examples", type=int, default=float('inf'), help="optionally cap the number of samples for evaluation. Defaults to lowest number of samples in test sets.")
parser.add_argument("--device", type=str, default="auto", help="")
parser.add_argument("--outdir", type=str, default="results/gradient-based-ood", help="")
parser.add_argument("--n_model", type=int, default=1, help="which model to use should be (1-2-3-4). Refers only to the new one I've trained")
parser.add_argument("--ood_data", type=str, default='svhn', help="ood dataset")


args = parser.parse_args()


oodd.utils.set_seed(args.seed)
print(args.model_dataset)
print(args.ood_data)
print(args.n_model)
######
#
# useful functions
######

def to_vector(tensors: List[torch.Tensor]):
    return torch.cat([t.flatten() for t in tensors])


def get_parameters(model, n_stages_skip=0, generative_parameters_only=False, inference_parameters_only=False):
    if n_stages_skip == 0:
        if generative_parameters_only:
            parameters = model.get_generative_parameters()
        elif inference_parameters_only:
            parameters = model.get_inference_parameters()
        else:
            parameters = model.parameters()
    else:
        if generative_parameters_only:
            stages = model.get_generative_parameters_per_stage()[n_stages_skip:]
            parameters = (p for s in stages for p in s)
        elif inference_parameters_only:
            stages = model.get_inference_parameters_per_stage()[n_stages_skip:]
            parameters = (p for s in stages for p in s)
        else:
            stages = model.get_parameters_per_stage()[n_stages_skip:]
            parameters = (p for s in stages for p in s)

    return (p for p in parameters if p.grad is not None)


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

saving_directory = "additional_models/estimated_quantities_model_trained_on_" + args.model_dataset.lower() + "_dequantized_" + str(args.dequantized) + '_n_vae_bound_computation_' + str(args.n_samples) + '_model_' + str(args.n_model) + '_ood_data_' + args.ood_data.lower() + '/'
os.makedirs(saving_directory + 'distributions', exist_ok=True)
os.makedirs(saving_directory + 'results', exist_ok=True)

device = oodd.utils.get_device() if args.device == "auto" else torch.device(args.device)
LOGGER.info("Device %s", device)
print(device)

# MAYBE I SHOULD CONSIDER THE PAPER MODELS AS A STARTING POINT
# let's try to understand how to load the model
trained_model_directory = 'models/'
print(args.dequantized)
print(args.n_posterior_samples)
if args.model_dataset.lower() == 'fashionmnist':
    if args.dequantized == 1:
        dataset_dir = 'FashionMNISTDequantized-2021-09-10-11-51-08.466465/'

        # I should load the datasets here, maybe it's better
        train_in, valid_in, test_in  = get_FashionMNIST_for_training(transformation='dequantized')
        test_ood = get_MNIST_test_set(transformation='dequantized')

        LOGGER.info("We are training on FashionMNISTDequantized (IN) and testing on MNISTDequantized (OOD)")
    else:

        train_in, valid_in, test_in = get_FashionMNIST_for_training(transformation='binarized')
        test_ood = get_MNIST_test_set(transformation='binarized')

        LOGGER.info("We are training on FashionMNISTBinarized (IN) and testing on MNISTBinarized (OOD)")

        if args.n_model == 1:
            dataset_dir = 'FashionMNISTBinarized-2021-12-30-14-46-47.317221/'
        elif args.n_model == 2:
            dataset_dir = 'FashionMNISTBinarized-2021-12-30-15-00-00.333292/'
        elif args.n_model == 3:
            dataset_dir = 'FashionMNISTBinarized-2021-12-30-15-00-39.680927/'
        elif args.n_model == 4:
            dataset_dir = 'FashionMNISTBinarized-2021-12-30-15-01-11.857973/'
        elif args.n_model == 10:
            # we use the original model we used in the paper
            trained_model_directory = 'trained_models/'
            dataset_dir = 'FashionMNISTBinarized-2021-09-10-18-14-56.756176/'
        else:
            raise ValueError('Wrong number for args.n_model')

    model_state_dict_path = trained_model_directory + dataset_dir + 'model_state_dict.pt'
    model_kwargs_path = trained_model_directory + dataset_dir + 'model_kwargs.pt'
    model_class_names_path = trained_model_directory + dataset_dir + 'model_class_name.pt'
    datamodule_config_path = trained_model_directory + dataset_dir + 'datamodule_config.pt'
    evaluator_path = trained_model_directory + dataset_dir + 'evaluator.pt'

elif args.model_dataset.lower() == 'mnist':
    trained_model_directory = 'trained_models/'
    if args.dequantized == 1:
        dataset_dir = 'MNISTDequantized-2021-09-10-11-51-29.630627/'

        # I should load the datasets here, maybe it's better
        train_in, valid_in, test_in = get_MNIST_for_training(transformation='dequantized')
        test_ood = get_FashionMNIST_test_set(transformation='dequantized')

        LOGGER.info("We are training on MNISTDequantized (IN) and testing on FashionMNISTDequantized (OOD)")
    else:
        dataset_dir = 'MNISTBinarized-2021-09-10-18-15-46.215983/'

        train_in, valid_in, test_in = get_MNIST_for_training(transformation='binarized')
        test_ood = get_FashionMNIST_test_set(transformation='binarized')

        LOGGER.info("We are training on MNISTBinarized (IN) and testing on FashionMNISTBinarized (OOD)")

    model_state_dict_path = trained_model_directory + dataset_dir + 'model_state_dict.pt'
    model_kwargs_path = trained_model_directory + dataset_dir + 'model_kwargs.pt'
    model_class_names_path = trained_model_directory + dataset_dir + 'model_class_name.pt'
    datamodule_config_path = trained_model_directory + dataset_dir + 'datamodule_config.pt'
    evaluator_path = trained_model_directory + dataset_dir + 'evaluator.pt'

elif args.model_dataset.lower() == 'cifar10':
    if args.dequantized == 1:
        if args.n_model == 1:
            dataset_dir = 'CIFAR10Dequantized-2021-12-21-15-00-15.459319/'
        elif args.n_model == 2:
            dataset_dir = 'CIFAR10Dequantized-2021-12-21-15-04-21.014782/'
        elif args.n_model == 3:
            dataset_dir = 'CIFAR10Dequantized-2021-12-21-15-08-51.135851/'
        elif args.n_model == 4:
            dataset_dir = 'CIFAR10Dequantized-2021-12-21-15-14-24.353860/'
        elif args.n_model == 5:
            dataset_dir = 'CIFAR10Dequantized-2022-01-26-22-51-32.123272/'
        elif args.n_model == 10:
            # we use the original model we used in the paper
            trained_model_directory = 'trained_models/'
            dataset_dir = 'CIFAR10Dequantized-2021-09-10-11-21-53.872137/'
        else:
            raise ValueError('Wrong number for args.n_model')

        # load the data we need
        train_in, valid_in, test_in = get_CIFAR10_for_training(transformation='dequantized')

        if args.ood_data.lower() == 'svhn':
            test_ood = get_SVHN_test_set(transformation='dequantized')
        elif args.ood_data.lower() == 'celeba':
            test_ood = get_celeba_test_set(transformation='dequantized')
        else:
            test_ood = get_CIFAR100_test_set(transformation='dequantized')
        # test_ood = get_CIFAR100_test_set(transformation='dequantized')
        # here I have to load CelebA
        # test_ood = get_celeba_test_set(transformation='dequantized')

    else:
        raise Exception(f'We only have trained on dequantized data, therefore use {args.dequantized}=True')

    model_state_dict_path = trained_model_directory + dataset_dir + 'model_state_dict.pt'
    model_kwargs_path = trained_model_directory + dataset_dir + 'model_kwargs.pt'
    model_class_names_path = trained_model_directory + dataset_dir + 'model_class_name.pt'
    datamodule_config_path = trained_model_directory + dataset_dir + 'datamodule_config.pt'
    evaluator_path = trained_model_directory + dataset_dir + 'evaluator.pt'

elif args.model_dataset.lower() == 'svhn':
    trained_model_directory = 'trained_models/'
    if args.dequantized == 1:
        dataset_dir = 'SVHNDequantized-2021-09-10-11-16-40.259864/'

        # load the data we need
        train_in, valid_in, test_in = get_SVHN_for_training(transformation='dequantized')
        test_ood = get_CIFAR_test_set(transformation='dequantized')

        LOGGER.info("We are training on SVHNDequantized (IN) and testing on CIFARDequantized (OOD)")
    else:
        raise Exception(f'We only have trained on dequantized data, therefore use {args.dequantized}=True')

    model_state_dict_path = trained_model_directory + dataset_dir + 'model_state_dict.pt'
    model_kwargs_path = trained_model_directory + dataset_dir + 'model_kwargs.pt'
    model_class_names_path = trained_model_directory + dataset_dir + 'model_class_name.pt'
    datamodule_config_path = trained_model_directory + dataset_dir + 'datamodule_config.pt'
    evaluator_path = trained_model_directory + dataset_dir + 'evaluator.pt'

elif args.model_dataset.lower() == 'celeba':
    trained_model_directory = 'models/'
    if args.dequantized == 1:
        dataset_dir = 'CelebADequantized-2021-11-26-23-07-43.986092/'

        # load the data we need
        train_in, valid_in, test_in = get_celeba_for_training(transformation='dequantized')
        # test_ood = get_CIFAR_test_set(transformation='dequantized')
        test_ood = get_SVHN_test_set(transformation='dequantized')


        # LOGGER.info("We are training on CELEBADequantized (IN) and testing on CIFAR1'Dequantized (OOD)")
        LOGGER.info("We are training on CELEBADequantized (IN) and testing on SVHNDequantized (OOD)")
    else:
        raise Exception(f'We only have trained on dequantized data, therefore use {args.dequantized}=True')

    model_state_dict_path = trained_model_directory + dataset_dir + 'model_state_dict.pt'
    model_kwargs_path = trained_model_directory + dataset_dir + 'model_kwargs.pt'
    model_class_names_path = trained_model_directory + dataset_dir + 'model_class_name.pt'
    datamodule_config_path = trained_model_directory + dataset_dir + 'datamodule_config.pt'
    evaluator_path = trained_model_directory + dataset_dir + 'evaluator.pt'

model = getattr(oodd.models.dvae, 'VAE')
# model_args = {'input_shape': torch.Size([3, 32, 32]), 'likelihood_module': 'DiscretizedLogisticMixLikelihoodConv2d', 'config_deterministic': [[{'block': 'ResBlockConv2d', 'out_channels': 32, 'kernel_size': 5, 'stride': 1, 'weightnorm': True, 'gated': False}, {'block': 'ResBlockConv2d', 'out_channels': 32, 'kernel_size': 5, 'stride': 1, 'weightnorm': True, 'gated': False}, {'block': 'ResBlockConv2d', 'out_channels': 64, 'kernel_size': 5, 'stride': 2, 'weightnorm': True, 'gated': False}], [{'block': 'ResBlockConv2d', 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'weightnorm': True, 'gated': False}, {'block': 'ResBlockConv2d', 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'weightnorm': True, 'gated': False}, {'block': 'ResBlockConv2d', 'out_channels': 128, 'kernel_size': 3, 'stride': 2, 'weightnorm': True, 'gated': False}], [{'block': 'ResBlockConv2d', 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'weightnorm': True, 'gated': False}, {'block': 'ResBlockConv2d', 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'weightnorm': True, 'gated': False}, {'block': 'ResBlockConv2d', 'out_channels': 256, 'kernel_size': 3, 'stride': 2, 'weightnorm': True, 'gated': False}]], 'config_stochastic': [{'block': 'GaussianConv2d', 'latent_features': 8, 'weightnorm': True}, {'block': 'GaussianConv2d', 'latent_features': 16, 'weightnorm': True}, {'block': 'GaussianConv2d', 'latent_features': 32, 'weightnorm': True}], 'q_dropout': 0.0, 'p_dropout': 0.0, 'activation': 'ReLU', 'skip_stochastic': True, 'padded_shape': None}
# model_args = {'input_shape': torch.Size([1, 28, 28]), 'likelihood_module': 'DiscretizedLogisticLikelihoodConv2d', 'config_deterministic': [[{'block': 'ResBlockConv2d', 'out_channels': 64, 'kernel_size': 5, 'stride': 1, 'weightnorm': True, 'gated': False}, {'block': 'ResBlockConv2d', 'out_channels': 64, 'kernel_size': 5, 'stride': 1, 'weightnorm': True, 'gated': False}, {'block': 'ResBlockConv2d', 'out_channels': 64, 'kernel_size': 5, 'stride': 2, 'weightnorm': True, 'gated': False}], [{'block': 'ResBlockConv2d', 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'weightnorm': True, 'gated': False}, {'block': 'ResBlockConv2d', 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'weightnorm': True, 'gated': False}, {'block': 'ResBlockConv2d', 'out_channels': 64, 'kernel_size': 3, 'stride': 2, 'weightnorm': True, 'gated': False}], [{'block': 'ResBlockConv2d', 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'weightnorm': True, 'gated': False}, {'block': 'ResBlockConv2d', 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'weightnorm': True, 'gated': False}, {'block': 'ResBlockConv2d', 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'weightnorm': True, 'gated': False}]], 'config_stochastic': [{'block': 'GaussianConv2d', 'latent_features': 8, 'weightnorm': True}, {'block': 'GaussianDense', 'latent_features': 16, 'weightnorm': True}, {'block': 'GaussianDense', 'latent_features': 8, 'weightnorm': True}], 'q_dropout': 0.0, 'p_dropout': 0.0, 'activation': 'ReLU', 'skip_stochastic': True, 'padded_shape': None}
model = model(**torch.load(model_kwargs_path)['kwargs'])

# print(model)
# I should also load the weights
model.load_state_dict(torch.load(model_state_dict_path))
model = model.to(device)
criterion = oodd.losses.ELBO()
print(criterion)
model.eval()

############################################
############################################
#
#         STARTING COMPUTATIONS
#
#
############################################
n_samples = args.n_samples
LOGGER.info(f"We have loaded {args.model_dataset} with dequantization={args.dequantized}")
# LOGGER.info("Model:\n%s", model)

# now I have to compute the things in the training error
fim_diag = 0
grad_mean = 0
log_like_mean = 0
grad_norm_mean = 0
i = 0
model.zero_grad()
print('starting computation of things from training set')
for batch_idx, (input,_) in tqdm(enumerate(train_in), desc='Computation using the training'):
    for idx, img in enumerate(input):
        if args.model_dataset.lower() == 'fashionmnist' or args.model_dataset.lower() == 'mnist':
            example = img.reshape(1, 1, 28, 28)
        else:
            example = img.reshape(1, 3, 32, 32)

        example = example.to(device)
        model.load_state_dict((torch.load(model_state_dict_path)))
        model.eval()
        model.zero_grad()

        likelihood_data, stage_datas = model(example, n_posterior_samples=args.n_posterior_samples)

        kl_divergences = [
            stage_data.loss.kl_elementwise
            for stage_data in stage_datas
            if stage_data.loss.kl_elementwise is not None
        ]
        loss, elbo, likelihood, kl_divergences = criterion(
            likelihood_data.likelihood,
            kl_divergences,
            samples=args.n_posterior_samples,
            free_nats=0,
            beta=1,
            sample_reduction=oodd.utils.log_sum_exp,
            batch_reduction=None,
        )

        (-loss).backward()

        _gradient = get_gradient_vector(
            get_parameters(model, generative_parameters_only=args.generative_parameters_only,
                           inference_parameters_only=args.inference_parameters_only)).detach()

        # print(_gradient.shape)
        # log_like_mean = 1 / (i + 1) * (i * log_like_mean + likelihood_data.likelihood)
        log_like_mean = 1 / (i + 1) * (i * log_like_mean + elbo)
        grad_mean = 1 / (i + 1) * (i * grad_mean + _gradient)
        fim_diag = 1 / (i + 1) * (i * fim_diag + (_gradient)**2)
        grad_norm_mean = (torch.norm(_gradient) + i * grad_norm_mean) / (i + 1)

        i += 1

        model.zero_grad()

# now I can store everything I am computing
print('Storing estimated quantities')
torch.save(grad_mean, saving_directory + 'grad_mean_training_estm.pt')
torch.save(log_like_mean, saving_directory + 'log_like_mean_training_estm.pt')
torch.save(grad_norm_mean, saving_directory + 'grad_norm_mean_training_estm.pt')
torch.save(fim_diag, saving_directory + 'fim_diag_training_estm.pt')

#at this point I can get the reciprocal of the fim
fim_approx = (fim_diag + 1e-8)
fim_reciprocal = 1/fim_approx


# now I can compute the quantities on the validation set
print('Computing typicality on the validation set')
mmd_fisher_valid = []
mmd_typicality_valid = []
mmd_identity_valid = []
score_statistic_valid = []
score_statistic_valid2 = []

for batch_idx, (input,_) in tqdm(enumerate(valid_in), desc='validation computation'):
    # print('We are evaluating the Validation set, we are at {}/{}'.format(batch_idx, valid_tot_batches))
    for idx, img in enumerate(input):
        if args.model_dataset.lower() == 'fashionmnist' or args.model_dataset.lower() == 'mnist':
            example = img.reshape(1, 1, 28, 28)
        else:
            example = img.reshape(1, 3, 32, 32)

        # example = example.to(device)
        model.load_state_dict((torch.load(model_state_dict_path)))
        model.eval()
        model.zero_grad()

        # I should create the batch using my example
        batch = example.repeat(args.n_samples, 1, 1, 1)
        batch = batch.to(device)

        likelihood_data, stage_datas = model(batch, n_posterior_samples=args.n_posterior_samples)

        kl_divergences = [
            stage_data.loss.kl_elementwise
            for stage_data in stage_datas
            if stage_data.loss.kl_elementwise is not None
        ]

        loss, elbo, likelihood, kl_divergences = criterion(
            likelihood_data.likelihood,
            kl_divergences,
            samples=args.n_posterior_samples,
            free_nats=0,
            beta=1,
            sample_reduction=oodd.utils.log_sum_exp,
            batch_reduction=None,
        )

        (-loss.mean()).backward()

        final_grad = get_gradient_vector(
            get_parameters(model, generative_parameters_only=args.generative_parameters_only,
                           inference_parameters_only=args.inference_parameters_only)).detach()

        # print(elbo.shape)
        final_elbo = elbo.mean()

        # now I can compute everything using the average we have just computed
        # and compute the mmd fisher for the example
        mmd_fisher_valid.append(torch.norm(torch.sqrt(fim_reciprocal) * (grad_mean - final_grad)).item())
        # mmd_typicality_valid.append(torch.norm(log_like_mean - likelihood_data.likelihood).item())
        mmd_typicality_valid.append(torch.norm(log_like_mean - final_elbo).item())
        mmd_identity_valid.append(torch.norm(torch.ones_like(fim_reciprocal) * (grad_mean - final_grad)).item())
        score_statistic_valid2.append(torch.norm(torch.sqrt(fim_reciprocal) * final_grad).item())
        score_statistic_valid.append((final_grad * fim_reciprocal).dot(final_grad).item())

        model.zero_grad()


# now I should store this
print('Storing the validation statistics')
torch.save(mmd_fisher_valid, saving_directory + 'distributions/mmd_fisher_valid.pt')
torch.save(mmd_typicality_valid, saving_directory + 'distributions/mmd_typicality_valid.pt')
torch.save(mmd_identity_valid, saving_directory + 'distributions/mmd_identity_valid.pt')
torch.save(score_statistic_valid, saving_directory + 'distributions/score_statistic_valid.pt')
torch.save(score_statistic_valid2, saving_directory + 'distributions/score_statistic_valid2.pt')

#### ECDF
# now I can compute the empirical CDF of these values
cdf_fisher = ECDF(mmd_fisher_valid)
cdf_typicality = ECDF(mmd_typicality_valid)
cdf_identity = ECDF(mmd_identity_valid)
cdf_score = ECDF(score_statistic_valid)
cdf_score2 = ECDF(score_statistic_valid2)

# now I have to do the same also for the two test sets
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
score_statistic_in2 = []
p_values_score_in2 = []
print('starting evaluation IN-test set')
for batch_idx, (batch,_) in tqdm(enumerate(test_in), desc='In-distribution test set'):
    for idx, example in enumerate(batch):
        if args.model_dataset.lower() == 'fashionmnist' or args.model_dataset.lower() == 'mnist':
            example = example.reshape(1, 1, 28, 28)
        else:
            example = example.reshape(1, 3, 32, 32)

        model.load_state_dict((torch.load(model_state_dict_path)))
        model.eval()
        model.zero_grad()

        # I should create the batch using my example
        batch = example.repeat(args.n_samples, 1, 1, 1)
        batch = batch.to(device)

        likelihood_data, stage_datas = model(batch, n_posterior_samples=args.n_posterior_samples)

        kl_divergences = [
            stage_data.loss.kl_elementwise
            for stage_data in stage_datas
            if stage_data.loss.kl_elementwise is not None
        ]

        loss, elbo, likelihood, kl_divergences = criterion(
            likelihood_data.likelihood,
            kl_divergences,
            samples=args.n_posterior_samples,
            free_nats=0,
            beta=1,
            sample_reduction=oodd.utils.log_sum_exp,
            batch_reduction=None,
        )

        (-loss.mean()).backward()

        final_grad = get_gradient_vector(
            get_parameters(model, generative_parameters_only=args.generative_parameters_only,
                           inference_parameters_only=args.inference_parameters_only)).detach()

        # print(elbo.shape)
        final_elbo = elbo.mean()

        # now I can compute the typicality test and p-values
        # log_like_in.append(likelihood_data.likelihood.item())
        log_like_in.append(final_elbo.item())
        grad_norm_in.append(torch.norm(final_grad).item())

        _mmd_fisher_score = torch.norm(torch.sqrt(fim_reciprocal) * (grad_mean - final_grad)).item()
        mmd_fisher_in.append(_mmd_fisher_score)
        p_values_fisher_in.append(1 - cdf_fisher(_mmd_fisher_score))

        # _typicality_score = torch.norm(log_like_mean - likelihood_data.likelihood).item()
        _typicality_score = torch.norm(log_like_mean - final_elbo).item()
        mmd_typicality_in.append(_typicality_score)
        p_values_typicality_in.append(1 - cdf_typicality(_typicality_score))

        _mmd_identity_score = torch.norm(torch.ones_like(fim_reciprocal) * (grad_mean - final_grad)).item()
        mmd_identity_in.append(_mmd_identity_score)
        p_values_identity_in.append(1 - cdf_identity(_mmd_identity_score))

        # _fisher_score = torch.norm(torch.sqrt(fim_reciprocal) * _grad0).item()
        _fisher_score = (final_grad * fim_reciprocal).dot(final_grad).item()
        score_statistic_in.append(_fisher_score)
        p_values_score_in.append(1 - cdf_score(_fisher_score))

        _fisher_score2 = torch.norm(torch.sqrt(fim_reciprocal) * final_grad).item()
        # _fisher_score = (final_grad * fim_reciprocal).dot(final_grad).item()
        score_statistic_in2.append(_fisher_score2)
        p_values_score_in2.append(1 - cdf_score2(_fisher_score2))

        model.zero_grad()


# now we can compute the statistics on the OOD data
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
score_statistic_ood2 = []
p_values_score_ood2 = []
print('starting evaluation on the OOD test set')
for batch_idx, (batch,_) in tqdm(enumerate(test_ood), desc='OOD dataset computation'):
    for idx, example in enumerate(batch):
        if args.model_dataset.lower() == 'fashionmnist' or args.model_dataset.lower() == 'mnist':
            example = example.reshape(1, 1, 28, 28)
        else:
            example = example.reshape(1, 3, 32, 32)

        model.load_state_dict((torch.load(model_state_dict_path)))
        model.eval()
        # I should create the batch using my example
        batch = example.repeat(args.n_samples, 1, 1, 1)
        batch = batch.to(device)

        likelihood_data, stage_datas = model(batch, n_posterior_samples=args.n_posterior_samples)

        kl_divergences = [
            stage_data.loss.kl_elementwise
            for stage_data in stage_datas
            if stage_data.loss.kl_elementwise is not None
        ]

        loss, elbo, likelihood, kl_divergences = criterion(
            likelihood_data.likelihood,
            kl_divergences,
            samples=args.n_posterior_samples,
            free_nats=0,
            beta=1,
            sample_reduction=oodd.utils.log_sum_exp,
            batch_reduction=None,
        )

        (-loss.mean()).backward()

        final_grad = get_gradient_vector(
            get_parameters(model, generative_parameters_only=args.generative_parameters_only,
                           inference_parameters_only=args.inference_parameters_only)).detach()

        # print(elbo.shape)
        final_elbo = elbo.mean()


        # now I can compute the typicality test and p-values
        # log_like_ood.append(likelihood_data.likelihood.item())
        log_like_ood.append(final_elbo.item())
        grad_norm_ood.append(torch.norm(final_grad).item())

        _mmd_fisher_score = torch.norm(torch.sqrt(fim_reciprocal) * (grad_mean - final_grad)).item()
        mmd_fisher_ood.append(_mmd_fisher_score)
        p_values_fisher_ood.append(1 - cdf_fisher(_mmd_fisher_score))

        # _typicality_score = torch.norm(log_like_mean - likelihood_data.likelihood).item()
        _typicality_score = torch.norm(log_like_mean - final_elbo).item()
        mmd_typicality_ood.append(_typicality_score)
        p_values_typicality_ood.append(1 - cdf_typicality(_typicality_score))

        _mmd_identity_score = torch.norm(torch.ones_like(fim_reciprocal) * (grad_mean - final_grad)).item()
        mmd_identity_ood.append(_mmd_identity_score)
        p_values_identity_ood.append(1 - cdf_identity(_mmd_identity_score))

        # _fisher_score = torch.norm(torch.sqrt(fim_reciprocal) * _grad0).item()
        _fisher_score = (final_grad * fim_reciprocal).dot(final_grad).item()
        score_statistic_ood.append(_fisher_score)
        p_values_score_ood.append(1 - cdf_score(_fisher_score))

        _fisher_score2 = torch.norm(torch.sqrt(fim_reciprocal) * final_grad).item()
        # _fisher_score = (final_grad * fim_reciprocal).dot(final_grad).item()
        score_statistic_ood2.append(_fisher_score2)
        p_values_score_ood2.append(1 - cdf_score2(_fisher_score2))

        model.zero_grad()


# NOW THAT I HAVE EVERYTHING I CAN STORE EVERYTHING

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

with open(saving_directory + 'results/in_dist_score_statistic2.pickle', 'wb') as handle:
    pkl.dump(score_statistic_in2, handle, protocol=pkl.HIGHEST_PROTOCOL)

with open(saving_directory + 'results/in_dist_p_values_score_stat2.pickle', 'wb') as handle:
    pkl.dump(p_values_score_in2, handle, protocol=pkl.HIGHEST_PROTOCOL)



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

with open(saving_directory + 'results/ood_dist_score_statistic2.pickle', 'wb') as handle:
    pkl.dump(score_statistic_ood2, handle, protocol=pkl.HIGHEST_PROTOCOL)

with open(saving_directory + 'results/ood_dist_p_values_score_stat.pickle', 'wb') as handle:
    pkl.dump(p_values_score_ood2, handle, protocol=pkl.HIGHEST_PROTOCOL)

# I can also already compute the AUROC scores and printing them
# on the logger

from sklearn.metrics import roc_auc_score
from scipy.stats import chi2

p_values_typicality_in = np.array(p_values_typicality_in)
p_values_score_in = np.array(p_values_score_in)

p_values_typicality_ood = np.array(p_values_typicality_ood)
p_values_score_ood = np.array(p_values_score_ood)


# I also want to check that the two way of computing the score statistic leads to
# the same result in terms of AUROC
p_values_score_in2 = np.array(p_values_score_in2)
p_values_score_ood2 = np.array(p_values_score_ood2)


y_ood = [1] * len(p_values_typicality_ood)
y_in = [0] * len(p_values_typicality_in)
y_true = y_ood + y_in
scores = np.concatenate((p_values_typicality_ood,p_values_typicality_in))
print('TYPICALITY AUROC: ', 1-roc_auc_score(y_true, scores))
LOGGER.info(f"TYPICALITY AUROC: {1-roc_auc_score(y_true, scores)}")



scores = np.concatenate((p_values_score_ood,p_values_score_in))
print('SCORE AUROC: ', 1-roc_auc_score(y_true, scores))
LOGGER.info(f"SCORE AUROC: {1-roc_auc_score(y_true, scores)}")

scores = np.concatenate((p_values_score_ood2,p_values_score_in2))
print('SCORE2 AUROC: ', 1-roc_auc_score(y_true, scores))
LOGGER.info(f"SCORE2 AUROC: {1-roc_auc_score(y_true, scores)}")

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

print('HARMONIC MEAN COMBINATION AUROC: ', 1-roc_auc_score(y_true, scores))

# I want to compute also the additional results

log_like_in = np.array(log_like_in)
log_like_ood = np.array(log_like_ood)
scores = np.concatenate((log_like_ood,log_like_in))
print('LOG-LIKE AUROC: ', 1-roc_auc_score(y_true, scores))

