import torch
import os
import numpy as np
import torch.nn as nn
import torchvision
from torch.distributions import Categorical, MixtureSameFamily, Normal, Independent
from torchvision import datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from model import trained_PPCA, trained_PPCA2
from tqdm import tqdm

from statsmodels.distributions.empirical_distribution import ECDF
import pickle as pkl

class Scale(nn.Module):
    def __init__(self, a=None, b=None, min_val=None, max_val=None):
        """Scale an input to be in [a, b] by normalizing with data min and max values"""
        super().__init__()
        assert (a is not None) == (b is not None), "must set both a and b or neither"
        self.a = a
        self.b = b
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        x_min = self.min_val if self.min_val is not None else x.min()
        x_max = self.max_val if self.max_val is not None else x.max()
        x_scaled = (x - x_min) / (x_max - x_min)
        if self.a is None:
            return x_scaled

        return self.a + x_scaled * (self.b - self.a)

class Dequantize(nn.Module):
    """Dequantize a quantized data point by adding uniform noise.

    Sppecifically, assume the quantized data is x in {0, 1, 2, ..., D} for some D e.g. 255 for int8 data.
    Then, the transformation is given by definition of the dequantized data z as

        z = x + u
        u ~ U(0, 1)

    where u is sampled uniform noise of same shape as x.

    The dequantized data is in the continuous interval [0, D + 1]

    If the value is to scaled subsequently, the maximum value attainable is hence D + 1 due to the uniform noise.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x + torch.rand_like(x)


# set up the seeds
np.random.seed(1)
torch.manual_seed(1)

# define the transformation
batch_size = 10

# I am trusting this transformation

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        Scale(a=0, b=255, min_val=0, max_val=1),  # Scale to [0, 255]
        Dequantize(),  # Add U(0, 1) noise, becomes [0, 256]
        Scale(a=0, b=1, min_val=0, max_val=256),  # Scale to [0, 1]
        torchvision.transforms.Lambda(lambda x: torch.flatten(x)),
    ]
)

# load the dataset and split it in train/validation/test
dataset = datasets.FashionMNIST('data/', download=True,
                             train=True, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True)

test_set = datasets.FashionMNIST('data/', train=False, download=True,
                          transform=transform)

test, valid = torch.utils.data.random_split(test_set, [7000, 3000],
                                            generator=torch.Generator().manual_seed(1))

valid_loader = torch.utils.data.DataLoader(valid,
                                            batch_size=batch_size,
                                            shuffle=False)

test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                          shuffle=False)

test_ood = torch.utils.data.DataLoader(datasets.MNIST('data/', download=True,
                             train=False, transform=transform), batch_size=batch_size,
                                            shuffle=False)


# to run sklearn methods I have to transform my dataset in numpy
dataset_numpy = []
for (batch,_) in train_loader:
  for img in batch:
    dataset_numpy.append(img.numpy())

dataset_numpy = np.array(dataset_numpy)
print('Training set example: ', dataset_numpy.shape)

# plt.imshow(dataset_numpy[0].reshape(28,28), cmap='gray')

# I need also the validation set now
valid_numpy = []
for (batch,_) in valid_loader:
  for img in batch:
    valid_numpy.append(img.numpy())

valid_numpy = np.array(valid_numpy)
print('Validation set example: ', valid_numpy.shape)

# todo: normalize data? Usually we do this when doing PCA

# I can make the algorith choosing the latent dimension (or number of components)
pca = PCA(n_components='mle', svd_solver='full')
pca.fit(dataset_numpy)

print(f'We are using {pca.n_components_} components')
# now we have our PCA fitted and I can retrieve the quantities I need
latent_dim = pca.n_components_
n_features = dataset_numpy[0].shape[0]
var = pca.noise_variance_
W = pca_W = pca.components_.T.dot(np.sqrt(np.diag(pca.explained_variance_) - pca.noise_variance_ * np.eye(pca.n_components_)))

# now I can use my wrap-class
device = 'cuda' if torch.cuda.is_available() else 'cpu'

W_torch = torch.from_numpy(W).float().to(device)
log_var = torch.tensor([np.log(var)]).float().to(device)
training_mean = torch.from_numpy(np.mean(dataset_numpy,axis = 0)).to(device)


ppca = trained_PPCA2(n_features, latent_dim, W_torch, log_var, training_mean, device)



print('Security check on the model parameters')
for name, param in ppca.named_parameters():
    print(name)
    print(param.shape)

# saving directory
saving_directory = "estimated_quantities_ppca_trained_on_fashionmnist_tested_on_mnist_latent_" + str(latent_dim ) + "/"
os.makedirs(saving_directory + 'distributions', exist_ok=True)
os.makedirs(saving_directory + 'results', exist_ok=True)
os.makedirs(saving_directory + 'model', exist_ok=True)

# here maybe I should store the model, because otherwise I cannot retrieve it
torch.save(ppca, saving_directory + 'model/model.pt')

ppca = ppca.to(device)
ppca.eval()

fim_diag = 0
grad_mean = 0
log_like_mean = 0
grad_norm_mean = 0
i = 0
print('starting computation of things from training set')
for batch_idx, (input,_) in tqdm(enumerate(train_loader), desc='Computation using the training'):
    for idx, img in enumerate(input):
        img = img.to(device)
        # net.load_state_dict((torch.load(checkpoint['net'])))
        ppca.eval()
        ppca.zero_grad()

        # I have to compute the log-likelihood of the single example
        log_like = ppca.compute_log_likelihood(img)

        log_like.backward()

        # I have to collect the gradient now
        grad = torch.nn.utils.parameters_to_vector([p.grad for p in ppca.parameters() if p.grad is not None]).detach()

        log_like_mean = 1 / (i + 1) * (i * log_like_mean + log_like.detach())
        grad_mean = 1 / (i + 1) * (i * grad_mean + grad)
        fim_diag = 1 / (i + 1) * (i * fim_diag + (grad) ** 2)
        grad_norm_mean = (torch.norm(grad) + i * grad_norm_mean) / (i + 1)

        i += 1
        ppca.zero_grad()

print('Storing estimated quantities')
torch.save(grad_mean, saving_directory + 'grad_mean_training_estm.pt')
torch.save(log_like_mean, saving_directory + 'log_like_mean_training_estm.pt')
torch.save(grad_norm_mean, saving_directory + 'grad_norm_mean_training_estm.pt')
torch.save(fim_diag, saving_directory + 'fim_diag_training_estm.pt')

# at this point I can get the reciprocal of the fim
fim_approx = (fim_diag + 1e-8)
fim_reciprocal = 1/fim_approx


print('Computing statistics on the validation set')
mmd_fisher_valid = []
mmd_typicality_valid = []
mmd_identity_valid = []
score_statistic_valid = []
grad_norm_valid = []
likelihood_valid = []

for batch_idx, (input,_) in tqdm(enumerate(valid_loader), desc='validation computation'):
    for idx, img in enumerate(input):
        img = img.to(device)

        ppca.eval()
        ppca.zero_grad()

        # I have to compute the log-likelihood of the single example
        log_like = ppca.compute_log_likelihood(img)

        log_like.backward()

        # I have to collect the gradient now
        grad = torch.nn.utils.parameters_to_vector([p.grad for p in ppca.parameters() if p.grad is not None]).detach()

        # and compute the mmd fisher for the example
        likelihood_valid.append((log_like.detach()).item())
        grad_norm_valid.append((torch.norm(grad)).item())
        mmd_fisher_valid.append(torch.norm(torch.sqrt(fim_reciprocal) * (grad_mean - grad)).item())
        # mmd_typicality_valid.append(torch.norm(log_like_mean - likelihood_data.likelihood).item())
        mmd_typicality_valid.append(torch.norm(log_like_mean - log_like.detach()).item())
        mmd_identity_valid.append(torch.norm(torch.ones_like(fim_reciprocal) * (grad_mean - grad)).item())
        score_statistic_valid.append(torch.norm(torch.sqrt(fim_reciprocal) * grad).item())

        ppca.zero_grad()

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
for batch_idx, (batch,_) in tqdm(enumerate(test_loader), desc='In-distribution test set'):
    for idx, img in enumerate(batch):
        img = img.to(device)

        ppca.eval()
        ppca.zero_grad()

        # I have to compute the log-likelihood of the single example
        log_like = ppca.compute_log_likelihood(img)

        log_like.backward()

        # I have to collect the gradient now
        grad = torch.nn.utils.parameters_to_vector([p.grad for p in ppca.parameters() if p.grad is not None]).detach()

        log_like_in.append((log_like).detach().item())
        grad_norm_in.append(torch.norm(grad).item())

        _mmd_fisher_score = torch.norm(torch.sqrt(fim_reciprocal) * (grad_mean - grad)).item()
        mmd_fisher_in.append(_mmd_fisher_score)
        p_values_fisher_in.append(1 - cdf_fisher(_mmd_fisher_score))

        # _typicality_score = torch.norm(log_like_mean - likelihood_data.likelihood).item()
        _typicality_score = torch.norm(log_like_mean - (log_like).detach()).item()
        mmd_typicality_in.append(_typicality_score)
        p_values_typicality_in.append(1 - cdf_typicality(_typicality_score))

        _mmd_identity_score = torch.norm(torch.ones_like(fim_reciprocal) * (grad_mean - grad)).item()
        mmd_identity_in.append(_mmd_identity_score)
        p_values_identity_in.append(1 - cdf_identity(_mmd_identity_score))

        _fisher_score = torch.norm(torch.sqrt(fim_reciprocal) * grad).item()
        score_statistic_in.append(_fisher_score)
        p_values_score_in.append(1 - cdf_score(_fisher_score))

        ppca.zero_grad()

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

for batch_idx, (batch,_) in tqdm(enumerate(test_ood), desc='OOD dataset computation'):
    for idx, img in enumerate(batch):
        img = img.to(device)

        ppca.eval()
        ppca.zero_grad()

        # I have to compute the log-likelihood of the single example
        log_like = ppca.compute_log_likelihood(img)

        log_like.backward()

        # I have to collect the gradient now
        grad = torch.nn.utils.parameters_to_vector([p.grad for p in ppca.parameters() if p.grad is not None]).detach()

        log_like_ood.append((log_like).detach().item())
        grad_norm_ood.append(torch.norm(grad).item())

        _mmd_fisher_score = torch.norm(torch.sqrt(fim_reciprocal) * (grad_mean - grad)).item()
        mmd_fisher_ood.append(_mmd_fisher_score)
        p_values_fisher_ood.append(1 - cdf_fisher(_mmd_fisher_score))

        # _typicality_score = torch.norm(log_like_mean - likelihood_data.likelihood).item()
        _typicality_score = torch.norm(log_like_mean - (log_like).detach().item()).item()
        mmd_typicality_ood.append(_typicality_score)
        p_values_typicality_ood.append(1 - cdf_typicality(_typicality_score))

        _mmd_identity_score = torch.norm(torch.ones_like(fim_reciprocal) * (grad_mean - grad)).item()
        mmd_identity_ood.append(_mmd_identity_score)
        p_values_identity_ood.append(1 - cdf_identity(_mmd_identity_score))

        _fisher_score = torch.norm(torch.sqrt(fim_reciprocal) * grad).item()
        score_statistic_ood.append(_fisher_score)
        p_values_score_ood.append(1 - cdf_score(_fisher_score))

        ppca.zero_grad()


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



scores = np.concatenate((p_values_score_ood,p_values_score_in))
print('SCORE AUROC: ', 1-roc_auc_score(y_true, scores))


# combined p-values
combined_p_values_in = -2 * (np.log(p_values_typicality_in + 1e-8) + np.log(p_values_score_in+ 1e-8))
combined_p_values_out = -2 * (np.log(p_values_typicality_ood + 1e-8) + np.log(p_values_score_ood + 1e-8))

scores = np.concatenate((combined_p_values_out,combined_p_values_in))
print('FISHER COMBINATION AUROC: ', roc_auc_score(y_true, scores))

# maybe I can also try to put the p-values together by using the harmonic stuff
harmonic_mean_in = (2 * p_values_typicality_in * p_values_score_in) / (p_values_typicality_in + p_values_score_in + 1e-8)
harmonic_mean_ood =(2 * p_values_typicality_ood * p_values_score_ood) / (p_values_typicality_ood + p_values_score_ood + 1e-8)

scores = np.concatenate((harmonic_mean_ood,harmonic_mean_in))

print(1-roc_auc_score(y_true, scores))







