'''
File used to compute the statistics described in the paper
"Model-agnostic out-of-distribution detection using combined statistical tests"
and to combine them using Fisher's method.

In this file, we are using a PixelCNN++ implementation from https://github.com/pclucas14/pixel-cnn-pp

In this case, we show for a model trained on CIFAR.
'''

import time
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, utils
from torch.utils.tensorboard import SummaryWriter
from utils import *
from model import *
from PIL import Image
from statsmodels.distributions.empirical_distribution import ECDF
import pickle as pkl
from tqdm import tqdm



parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-i', '--data_dir', type=str,
                    default='/scratch/fedbe/cifar10/', help='Location for the dataset')
parser.add_argument('-o', '--save_dir', type=str, default='models',
                    help='Location for parameter checkpoints and samples')
parser.add_argument('-d', '--dataset', type=str,
                    default='cifar', help='Can be either cifar|mnist|svhn')
parser.add_argument('-p', '--print_every', type=int, default=50,
                    help='how many iterations between print statements')
parser.add_argument('-t', '--save_interval', type=int, default=50,
                    help='Every how many epochs to write checkpoint/samples?')
parser.add_argument('-r', '--load_params', type=str, default=None,
                    help='Restore training from previous model checkpoint?')
# model
parser.add_argument('-q', '--nr_resnet', type=int, default=5,
                    help='Number of residual blocks per stage of the model')
parser.add_argument('-n', '--nr_filters', type=int, default=160,
                    help='Number of filters to use across the model. Higher = larger model.')
parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10,
                    help='Number of logistic components in the mixture. Higher = more flexible model')
parser.add_argument('-l', '--lr', type=float,
                    default=0.0002, help='Base learning rate')
parser.add_argument('-e', '--lr_decay', type=float, default=0.999995,
                    help='Learning rate decay, applied every step of the optimization')
parser.add_argument('-b', '--batch_size', type=int, default=64,
                    help='Batch size during training per GPU')
parser.add_argument('-x', '--max_epochs', type=int,
                    default=5000, help='How many epochs to run in total?')
parser.add_argument('-s', '--seed', type=int, default=1,
                    help='Random seed to use')
args = parser.parse_args()


# reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)

saving_directory = 'estimated_quantities_aistat_1079/'
os.makedirs(saving_directory + 'distributions', exist_ok=True)
os.makedirs(saving_directory + 'results', exist_ok=True)

sample_batch_size = 25
obs = (1, 28, 28) if 'fashion' in args.dataset else (3, 32, 32)
input_channels = obs[0]
rescaling     = lambda x : (x - .5) * 2.
rescaling_inv = lambda x : .5 * x  + .5
kwargs = {'num_workers':4, 'pin_memory':True, 'drop_last':True}
ds_transforms = transforms.Compose([transforms.ToTensor(), rescaling])

# load the datasets I need:
# - train FASHION-MNIST
# - valid FASHION-MNIST
# - test FASHION-MNIST
# - test MNIST


dataset = datasets.CIFAR10(args.data_dir, train=True,
                           download=True, transform=ds_transforms)  # todo: shuffle this?

train_loader_cifar = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, **kwargs)

test_set = datasets.CIFAR10(args.data_dir, train=False,
                            transform=ds_transforms)

test, valid = torch.utils.data.random_split(test_set, [7000, 3000],
                                            generator=torch.Generator().manual_seed(args.seed))

valid_loader_cifar = torch.utils.data.DataLoader(valid, batch_size=10, shuffle=True, **kwargs)

test_loader_cifar = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True, **kwargs)


test_loader_svhn = torch.utils.data.DataLoader(datasets.SVHN(args.data_dir, split='test',download=True,
                                                           transform=ds_transforms), batch_size=10,
                                          shuffle=True, **kwargs)

loss_op   = lambda real, fake : discretized_mix_logistic_loss(real, fake)
sample_op = lambda x : sample_from_discretized_mix_logistic(x, args.nr_logistic_mix)

params_path = 'models/pcnn_lr:0.00020_nr-resnet5_nr-filters160_dataser_cifar_1079.pth'

model = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters,
            input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix)

model = nn.DataParallel(model)
model.load_state_dict(torch.load(params_path))
model = model.cuda()
model.eval()

# now I have to estimate everything by using the training set
#### ESTIMATION using training set
fim_diag = 0
grad_mean = 0
log_like_mean = 0
grad_norm_mean = 0
i = 0
print('starting computation of things from training set')
for batch_idx, (input,_) in tqdm(enumerate(train_loader_cifar), desc='training comp'):
    for idx, img in enumerate(input):
        # model.zero_grad()
        example = img.reshape(1, 3, 32, 32)
        model.load_state_dict((torch.load(params_path)))
        model.eval()
        model.zero_grad()

        input = example.cuda()
        input_var = Variable(input)
        output = model(input_var)
        loss = loss_op(input_var, output)
        loss.backward()

        # get gradient as the first thing
        g = torch.nn.utils.parameters_to_vector(-p.grad for p in model.parameters()).detach()
        log_like_mean = (-loss.detach() + i*log_like_mean)/(i+1)
        fim_diag = ((g)**2 + i*fim_diag)/(i+1)
        grad_mean = ((g) + i*grad_mean)/(i+1)
        ## average gradient norm
        grad_norm_mean = (torch.norm(g) + i*grad_norm_mean)/(i+1)
        i+=1

        model.zero_grad()

## I want to store all the things we compute
print('Storing estimated quantities')
torch.save(grad_mean, saving_directory +'grad_mean_training_estm_no_eval.pt')
torch.save(log_like_mean, saving_directory +'log_like_mean_training_estm_no_eval.pt')
torch.save(grad_norm_mean, saving_directory +'grad_norm_mean_training_estm_no_eval.pt')
torch.save(fim_diag, saving_directory +'fim_diag_training_estm_no_eval.pt')

# at this point I can get the reciprocal of the fim
fim_approx = (fim_diag + 1e-8)
fim_reciprocal = 1/fim_approx

# now I can compute the quantities on the validation set
valid_tot_batches = int(len(valid_loader_cifar.dataset) / args.batch_size)
print('Computing typicality on the validation set')
#todo: I am calling them train, but they are validation
mmd_fisher_valid = []
mmd_typicality_valid = []
mmd_identity_valid = []
score_statistic_valid = []

for batch_idx, (input,_) in tqdm(enumerate(valid_loader_cifar), desc='valid'):
    # print('We are evaluating the VALIDATION set, we are at {}/{}'.format(batch_idx, valid_tot_batches))
    for idx, img in enumerate(input):
        # model.zero_grad()
        example = img.reshape(1, 3, 32, 32)
        model.load_state_dict((torch.load(params_path)))
        model.eval()
        model.zero_grad()

        input = example.cuda()
        input_var = Variable(input)
        output = model(input_var)
        loss = loss_op(input_var, output)
        loss.backward()

        # compute statistic for the training set
        # get gradient
        _grad0 = torch.nn.utils.parameters_to_vector(-p.grad for p in model.parameters()).detach()

        # and compute the mmd fisher for the example
        mmd_fisher_valid.append(torch.norm(torch.sqrt(fim_reciprocal) * (grad_mean - _grad0)).item())
        mmd_typicality_valid.append(torch.norm(log_like_mean - (-loss)).item())
        mmd_identity_valid.append(torch.norm(torch.ones_like(fim_reciprocal) * (grad_mean - _grad0)).item())
        score_statistic_valid.append(torch.norm(torch.sqrt(fim_reciprocal) * _grad0).item())

        model.zero_grad()

# store validation statistics
print('Storing the validation statistics')
torch.save(mmd_fisher_valid, saving_directory +'distributions/mmd_fisher_valid_no_eval.pt')
torch.save(mmd_typicality_valid,saving_directory + 'distributions/mmd_typicality_valid_no_eval.pt')
torch.save(mmd_identity_valid, saving_directory +'distributions/mmd_identity_valid_no_eval.pt')
torch.save(score_statistic_valid, saving_directory +'distributions/score_statistic_valid_no_eval.pt')

# now I can compute the distribution of these values
cdf_fisher = ECDF(mmd_fisher_valid)
cdf_typicality = ECDF(mmd_typicality_valid)
cdf_identity = ECDF(mmd_identity_valid)
cdf_score = ECDF(score_statistic_valid)

# loop over the two different test sets to compute the mmd fisher score for
# each example and the p-values
# FASHION-MNIST
#### ---- CIFAR
mmd_fisher_cifar = []
p_values_cifar_fisher = []
mmd_typicality_cifar = []
p_values_cifar_typicality = []
mmd_identity_cifar = []
p_values_cifar_identity = []
score_statistic_cifar = []
p_values_cifar_score = []
grad_norm_cifar = []
log_like_cifar = []
print('starting evaluation CIFAR')
cifar_tot_batches = int(len(test_loader_cifar.dataset) / args.batch_size)
for batch_idx, (batch,_) in tqdm(enumerate(test_loader_cifar), desc='test IN'):
    # if batch_idx % 10 ==0:
        # print('We are evaluating CIFAR10 test set, we are at {}/{}'.format(batch_idx, cifar_tot_batches))
    for idx, example in enumerate(batch):
        # model.zero_grad()
        example = example.reshape(1, 3, 32, 32)

        # I think we should load the weights again if we want to have
        # something meaningful for each example
        model.load_state_dict((torch.load(params_path)))
        # print('Cabin crew, boarding complete.')
        model.eval()
        model.zero_grad()
        # define a new optimizer
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.0002, momentum=0.9)
        input = example.cuda()
        input_var = Variable(input)
        output = model(input_var)
        loss = loss_op(input_var, output)
        # optimizer.zero_grad()
        loss.backward()

        # get the gradient
        _grad0 = torch.nn.utils.parameters_to_vector(-p.grad for p in model.parameters()).detach()

        grad_norm_cifar.append(torch.norm(_grad0).item())
        log_like_cifar.append((-loss).item())
        # now I can compute the typicality test and p-values
        _mmd_fisher_score = torch.norm(torch.sqrt(fim_reciprocal) * (grad_mean - _grad0)).item()
        mmd_fisher_cifar.append(_mmd_fisher_score)
        p_values_cifar_fisher.append(1-cdf_fisher(_mmd_fisher_score))

        _typicality_score = torch.norm(log_like_mean - (-loss)).item()
        mmd_typicality_cifar.append(_typicality_score)
        p_values_cifar_typicality.append(1 - cdf_typicality(_typicality_score))

        _mmd_identity_score = torch.norm(torch.ones_like(fim_reciprocal) * (grad_mean - _grad0)).item()
        mmd_identity_cifar.append(_mmd_identity_score)
        p_values_cifar_identity.append(1 - cdf_identity(_mmd_identity_score))

        _fisher_score = torch.norm(torch.sqrt(fim_reciprocal) * _grad0).item()
        score_statistic_cifar.append(_fisher_score)
        p_values_cifar_score.append(1 - cdf_score(_fisher_score))

        model.zero_grad()

#SVHN
svhn_tot_batches = int(len(test_loader_svhn.dataset) / args.batch_size)
mmd_fisher_svhn = []
p_values_fisher_svhn = []
mmd_typicality_svhn = []
p_values_svhn_typicality = []
mmd_identity_svhn = []
p_values_svhn_identity = []
score_statistic_svhn = []
p_values_svhn_score = []
grad_norm_svhn = []
log_like_svhn = []
print('starting evaluation SVHN')
for batch_idx, (batch, _) in tqdm(enumerate(test_loader_svhn),desc='test OOD'):
    # if batch_idx % 10 == 0:
    #     print('We are evaluating SVHN test set, we are at {}/{}'.format(batch_idx, svhn_tot_batches))
    for idx, example in enumerate(batch):
        # model.zero_grad()
        example = example.reshape(1, 3, 32, 32)

        # I think we should load the weights again if we want to have
        # something meaningful for each example
        model.load_state_dict((torch.load(params_path)))
        # print('Cabin crew, boarding complete.')
        model.eval()
        model.zero_grad()
        # define a new optimizer
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.0002, momentum=0.9)
        input = example.cuda()
        input_var = Variable(input)
        output = model(input_var)
        loss = loss_op(input_var, output)
        # optimizer.zero_grad()
        loss.backward()

        # get the gradient
        _grad0 = torch.nn.utils.parameters_to_vector(-p.grad for p in model.parameters()).detach()

        grad_norm_svhn.append(torch.norm(_grad0).item())
        log_like_svhn.append((-loss).item())
        # now I can compute the typicality test and p-values
        _mmd_fisher_score = torch.norm(torch.sqrt(fim_reciprocal) * (grad_mean - _grad0)).item()
        mmd_fisher_svhn.append(_mmd_fisher_score)
        p_values_fisher_svhn.append(1 - cdf_fisher(_mmd_fisher_score))

        _typicality_score = torch.norm(log_like_mean - (-loss)).item()
        mmd_typicality_svhn.append(_typicality_score)
        p_values_svhn_typicality.append(1 - cdf_typicality(_typicality_score))

        _mmd_identity_score = torch.norm(torch.ones_like(fim_reciprocal) * (grad_mean - _grad0)).item()
        mmd_identity_svhn.append(_mmd_identity_score)
        p_values_svhn_identity.append(1 - cdf_identity(_mmd_identity_score))

        _fisher_score = torch.norm(torch.sqrt(fim_reciprocal) * _grad0).item()
        score_statistic_svhn.append(_fisher_score)
        p_values_svhn_score.append(1 - cdf_score(_fisher_score))

        model.zero_grad()


# now I have to store everything
## -------------------- CIFAR test set -------------------
with open(saving_directory +'results/cifar_score_mmd_fisher_model_train_on_cifar_no_eval.pickle', 'wb') as handle:
    pkl.dump(mmd_fisher_cifar, handle, protocol=pkl.HIGHEST_PROTOCOL)

with open(saving_directory +'results/cifar_score_mmd_typicality_model_train_on_cifar_no_eval.pickle', 'wb') as handle:
    pkl.dump(mmd_typicality_cifar, handle, protocol=pkl.HIGHEST_PROTOCOL)

with open(saving_directory +'results/cifar_score_mmd_identity_model_train_on_cifar_no_eval.pickle', 'wb') as handle:
    pkl.dump(mmd_identity_cifar, handle, protocol=pkl.HIGHEST_PROTOCOL)

with open(saving_directory +'results/cifar_score_statistic_model_train_on_cifar_no_eval.pickle', 'wb') as handle:
    pkl.dump(score_statistic_cifar, handle, protocol=pkl.HIGHEST_PROTOCOL)

with open(saving_directory +'results/cifar_p_values_fisher_model_train_on_cifar_no_eval.pickle', 'wb') as handle:
    pkl.dump(p_values_cifar_fisher, handle, protocol=pkl.HIGHEST_PROTOCOL)

with open(saving_directory +'results/cifar_p_values_typicality_model_train_on_cifar_no_eval.pickle', 'wb') as handle:
    pkl.dump(p_values_cifar_typicality, handle, protocol=pkl.HIGHEST_PROTOCOL)

with open(saving_directory +'results/cifar_p_values_identity_model_train_on_cifar_no_eval.pickle', 'wb') as handle:
    pkl.dump(p_values_cifar_identity, handle, protocol=pkl.HIGHEST_PROTOCOL)

with open(saving_directory +'results/cifar_p_values_score_stat_model_train_on_cifar_no_eval.pickle', 'wb') as handle:
    pkl.dump(p_values_cifar_score, handle, protocol=pkl.HIGHEST_PROTOCOL)

with open(saving_directory +'results/cifar_grad_norm.pickle', 'wb') as handle:
    pkl.dump(grad_norm_cifar, handle, protocol=pkl.HIGHEST_PROTOCOL)

with open(saving_directory +'results/cifar_log_like.pickle', 'wb') as handle:
    pkl.dump(log_like_cifar, handle, protocol=pkl.HIGHEST_PROTOCOL)


## -------------------- SVHN test set -------------------
with open(saving_directory +'results/svhn_score_mmd_fisher_model_train_on_cifar_no_eval.pickle', 'wb') as handle:
    pkl.dump(mmd_fisher_svhn, handle, protocol=pkl.HIGHEST_PROTOCOL)

with open(saving_directory +'results/svhn_score_mmd_typicality_model_train_on_cifar_no_eval.pickle', 'wb') as handle:
    pkl.dump(mmd_typicality_svhn, handle, protocol=pkl.HIGHEST_PROTOCOL)

with open(saving_directory +'results/svhn_score_mmd_identity_model_train_on_cifar_no_eval.pickle', 'wb') as handle:
    pkl.dump(mmd_identity_svhn, handle, protocol=pkl.HIGHEST_PROTOCOL)

with open(saving_directory +'results/svhn_score_statistic_model_train_on_cifar_no_eval.pickle', 'wb') as handle:
    pkl.dump(score_statistic_svhn, handle, protocol=pkl.HIGHEST_PROTOCOL)

with open(saving_directory +'svhn_p_values_fisher_model_train_on_cifar_no_eval.pickle', 'wb') as handle:
    pkl.dump(p_values_fisher_svhn, handle, protocol=pkl.HIGHEST_PROTOCOL)

with open(saving_directory +'results/svhn_p_values_typicality_model_train_on_cifar_no_eval.pickle', 'wb') as handle:
    pkl.dump(p_values_svhn_typicality, handle, protocol=pkl.HIGHEST_PROTOCOL)

with open(saving_directory +'results/svhn_p_values_identity_model_train_on_cifar_no_eval.pickle', 'wb') as handle:
    pkl.dump(p_values_svhn_identity, handle, protocol=pkl.HIGHEST_PROTOCOL)

with open(saving_directory +'results/svhn_p_values_score_stat_model_train_on_cifar_no_eval.pickle', 'wb') as handle:
    pkl.dump(p_values_svhn_score, handle, protocol=pkl.HIGHEST_PROTOCOL)

with open(saving_directory +'results/svhn_grad_norm.pickle', 'wb') as handle:
    pkl.dump(grad_norm_svhn, handle, protocol=pkl.HIGHEST_PROTOCOL)

with open(saving_directory +'results/svhn_log_like.pickle', 'wb') as handle:
    pkl.dump(log_like_svhn, handle, protocol=pkl.HIGHEST_PROTOCOL)




from sklearn.metrics import roc_auc_score
from scipy.stats import chi2


print('CIFAR TRAINED DROPOUT modle trained by me')
p_values_typicality_in = np.array(p_values_cifar_typicality)
p_values_score_in = np.array(p_values_cifar_score)

p_values_typicality_ood = np.array(p_values_svhn_typicality)
p_values_score_ood = np.array(p_values_svhn_score)

y_ood = [1] * len(p_values_typicality_ood)
y_in = [0] * len(p_values_typicality_in)
y_true = y_ood + y_in
scores = np.concatenate((p_values_typicality_ood,p_values_typicality_in))
print('TYPICALITY AUROC: ', 1-roc_auc_score(y_true, scores))
# LOGGER.info(f"TYPICALITY AUROC: {1-roc_auc_score(y_true, scores)}")



scores = np.concatenate((p_values_score_ood,p_values_score_in))
print('SCORE AUROC: ', 1-roc_auc_score(y_true, scores))
# LOGGER.info(f"SCORE AUROC: {1-roc_auc_score(y_true, scores)}")


# combined p-values
combined_p_values_in = -2 * (np.log(p_values_typicality_in + 1e-8) + np.log(p_values_score_in+ 1e-8))
combined_p_values_out = -2 * (np.log(p_values_typicality_ood + 1e-8) + np.log(p_values_score_ood + 1e-8))

scores = np.concatenate((combined_p_values_out,combined_p_values_in))
print('FISHER COMBINATION AUROC: ', roc_auc_score(y_true, scores))
# LOGGER.info(f"FISHER COMBINATION AUROC: {roc_auc_score(y_true, scores)}")

# maybe I can also try to put the p-values together by using the harmonic stuff
harmonic_mean_in = (2 * p_values_typicality_in * p_values_score_in) / (p_values_typicality_in + p_values_score_in + 1e-8)
harmonic_mean_ood =(2 * p_values_typicality_ood * p_values_score_ood) / (p_values_typicality_ood + p_values_score_ood + 1e-8)

scores = np.concatenate((harmonic_mean_ood,harmonic_mean_in))

print(1-roc_auc_score(y_true, scores))
