"""
original code from
https://github.com/dhanajitb/GAIN-Pytorch

original paper
Pytorch implementation of the paper GAIN: Missing Data Imputation using Generative Adversarial Nets by Jinsung Yoon, James Jordon, Mihaela van der Schaar

modified by K.H.
Implemented as an imputer class

"""

# %% Packages
import torch
import numpy as np
import torch.nn.functional as F
from tqdm.notebook import tqdm_notebook as tqdm
import torch.nn as nn


class GAIN:
    def __init__(self,
                 use_gpu=True,
                 mb_size=16,
                 train_rate=0.95,
                 iteration=5000,
                 p_hint=0.9,
                 alpha=10,
                 batch_norm=False
                 ):

        self.train_rate = train_rate
        self.iteration = iteration
        self.p_hint = p_hint
        self.use_gpu = use_gpu
        self.alpha = alpha
        self.mb_size = mb_size

        # batch normalization: useless for most regression tasks
        self.batch_norm = batch_norm

    def fit_transform(self, dat_list):
        self._set_masking_array(dat_list)
        self._split()

        if self.batch_norm:
            self.bn = nn.BatchNorm1d(self.Dim)
            if self.use_gpu:
                self.bn = self.bn.cuda().double()

        self._init_theta()
        self.train()
        reconst_array = self._transform()
        final_reconst_array = (self.Data*self.Missing) + \
            (reconst_array*(1-self.Missing))[0]

        return final_reconst_array

    # prepare masking array
    def _set_masking_array(self, dat_list):
        # prepare masking array
        inp = np.array(dat_list)
        missing = np.array(inp)
        missing[np.where(missing == missing)] = 1
        missing[np.where(missing != missing)] = 0

        inp[np.where(inp != inp)] = 0

        self.Data = inp
        self.Missing = missing

    # split database
    def _split(self):
        self.No = len(self.Data)
        self.Dim = len(self.Data[0, :])
        self.H_Dim1 = self.Dim
        self.H_Dim2 = self.Dim

        self.idx = np.random.permutation(self.No)

        self.Train_No = int(self.No * self.train_rate)
        self.Test_No = self.No - self.Train_No

        # Train / Test Features
        self.trainX = self.Data[self.idx[:self.Train_No], :]
        self.testX = self.Data[self.idx[self.Train_No:], :]

        # Train / Test Missing Indicators
        self.trainM = self.Missing[self.idx[:self.Train_No], :]
        self.testM = self.Missing[self.idx[self.Train_No:], :]

    def _init_theta(self):
        Dim = self.Dim
        # %% 1. Discriminator
        if self.use_gpu is True:
            self.D_W1 = torch.tensor(xavier_init(
                [Dim*2, self.H_Dim1]), requires_grad=True, device="cuda")     # Data + Hint as inputs
            self.D_b1 = torch.tensor(
                np.zeros(shape=[self.H_Dim1]), requires_grad=True, device="cuda")

            self.D_W2 = torch.tensor(xavier_init(
                [self.H_Dim1, self.H_Dim2]), requires_grad=True, device="cuda")
            self.D_b2 = torch.tensor(
                np.zeros(shape=[self.H_Dim2]), requires_grad=True, device="cuda")

            self.D_W3 = torch.tensor(xavier_init(
                [self.H_Dim2, Dim]), requires_grad=True, device="cuda")
            # Output is multi-variate
            self.D_b3 = torch.tensor(
                np.zeros(shape=[Dim]), requires_grad=True, device="cuda")
        else:
            # Data + Hint as inputs
            self.D_W1 = torch.tensor(xavier_init(
                [Dim*2, self.H_Dim1]), requires_grad=True)
            self.D_b1 = torch.tensor(
                np.zeros(shape=[self.H_Dim1]), requires_grad=True)

            self.D_W2 = torch.tensor(xavier_init(
                [self.H_Dim1, self.H_Dim2]), requires_grad=True)
            self.D_b2 = torch.tensor(
                np.zeros(shape=[self.H_Dim2]), requires_grad=True)

            self.D_W3 = torch.tensor(xavier_init(
                [self.H_Dim2, Dim]), requires_grad=True)
            # Output is multi-variate
            self.D_b3 = torch.tensor(np.zeros(shape=[Dim]), requires_grad=True)

        self.theta_D = [self.D_W1, self.D_W2,
                        self.D_W3, self.D_b1, self.D_b2, self.D_b3]

        # %% 2. Generator
        if self.use_gpu is True:
            # Data + Mask as inputs (Random Noises are in Missing Components)
            self.G_W1 = torch.tensor(xavier_init(
                [Dim*2, self.H_Dim1]), requires_grad=True, device="cuda")
            self.G_b1 = torch.tensor(
                np.zeros(shape=[self.H_Dim1]), requires_grad=True, device="cuda")

            self.G_W2 = torch.tensor(xavier_init(
                [self.H_Dim1, self.H_Dim2]), requires_grad=True, device="cuda")
            self.G_b2 = torch.tensor(
                np.zeros(shape=[self.H_Dim2]), requires_grad=True, device="cuda")

            self.G_W3 = torch.tensor(xavier_init(
                [self.H_Dim2, Dim]), requires_grad=True, device="cuda")
            self.G_b3 = torch.tensor(
                np.zeros(shape=[Dim]), requires_grad=True, device="cuda")
        else:
            # Data + Mask as inputs (Random Noises are in Missing Components)
            self.G_W1 = torch.tensor(xavier_init(
                [Dim*2, self.H_Dim1]), requires_grad=True)
            self.G_b1 = torch.tensor(
                np.zeros(shape=[self.H_Dim1]), requires_grad=True)

            self.G_W2 = torch.tensor(xavier_init(
                [self.H_Dim1, self.H_Dim2]), requires_grad=True)
            self.G_b2 = torch.tensor(
                np.zeros(shape=[self.H_Dim2]), requires_grad=True)

            self.G_W3 = torch.tensor(xavier_init(
                [self.H_Dim2, Dim]), requires_grad=True)
            self.G_b3 = torch.tensor(np.zeros(shape=[Dim]), requires_grad=True)

        self.theta_G = [self.G_W1, self.G_W2,
                        self.G_W3, self.G_b1, self.G_b2, self.G_b3]

    # %% 1. Generator
    def generator(self, new_x, m):
        # Mask + Data Concatenate
        inputs = torch.cat(dim=1, tensors=[new_x, m])
        G_h1 = F.relu(torch.matmul(inputs, self.G_W1) + self.G_b1)
        G_h2 = F.relu(torch.matmul(G_h1, self.G_W2) + self.G_b2)
        # [0,1] normalized Output
        G_prob = torch.sigmoid(torch.matmul(G_h2, self.G_W3) + self.G_b3)

        if self.batch_norm:
            G_prob = self.bn(G_prob)

        return G_prob

    # %% 2. Discriminator
    def discriminator(self, new_x, h):
        # Hint + Data Concatenate
        inputs = torch.cat(dim=1, tensors=[new_x, h])
        D_h1 = F.relu(torch.matmul(inputs, self.D_W1) + self.D_b1)
        D_h2 = F.relu(torch.matmul(D_h1, self.D_W2) + self.D_b2)
        D_logit = torch.matmul(D_h2, self.D_W3) + self.D_b3
        D_prob = torch.sigmoid(D_logit)  # [0,1] Probability Output

        return D_prob

    def discriminator_loss(self, M, New_X, H):
        # Generator
        G_sample = self.generator(New_X, M)
        # Combine with original data
        Hat_New_X = New_X * M + G_sample * (1-M)

        # Discriminator
        D_prob = self.discriminator(Hat_New_X, H)

        # %% Loss
        D_loss = -torch.mean(M * torch.log(D_prob + 1e-8) +
                             (1-M) * torch.log(1. - D_prob + 1e-8))
        return D_loss

    def generator_loss(self, X, M, New_X, H):
        # %% Structure
        # Generator
        G_sample = self.generator(New_X, M)

        # Combine with original data
        Hat_New_X = New_X * M + G_sample * (1-M)

        # Discriminator
        D_prob = self.discriminator(Hat_New_X, H)

        # %% Loss
        G_loss1 = -torch.mean((1-M) * torch.log(D_prob + 1e-8))
        MSE_train_loss = torch.mean(
            (M * New_X - M * G_sample)**2) / torch.mean(M)

        G_loss = G_loss1 + self.alpha * MSE_train_loss

        # %% MSE Performance metric
        MSE_test_loss = torch.mean(
            ((1-M) * X - (1-M)*G_sample)**2) / torch.mean(1-M)
        return G_loss, MSE_train_loss, MSE_test_loss

    def test_loss(self, X, M, New_X):
        # %% Structure
        # Generator
        G_sample = self.generator(New_X, M)

        # %% MSE Performance metric
        MSE_test_loss = torch.mean(
            ((1-M) * X - (1-M)*G_sample)**2) / torch.mean(1-M)
        return MSE_test_loss, G_sample

    def train(self):

        # train
        optimizer_D = torch.optim.Adam(params=self.theta_D)
        optimizer_G = torch.optim.Adam(params=self.theta_G)

        # %% Start Iterations
        for it in tqdm(range(self.iteration)):

            # %% Inputs
            mb_idx = sample_idx(self.Train_No, self.mb_size)
            X_mb = self.trainX[mb_idx, :]

            Z_mb = sample_Z(self.mb_size, self.Dim)
            M_mb = self.trainM[mb_idx, :]
            H_mb1 = sample_M(self.mb_size, self.Dim, 1-self.p_hint)
            H_mb = M_mb * H_mb1

            New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb  # Missing Data Introduce

            if self.use_gpu is True:
                X_mb = torch.tensor(X_mb, device="cuda")
                M_mb = torch.tensor(M_mb, device="cuda")
                H_mb = torch.tensor(H_mb, device="cuda")
                New_X_mb = torch.tensor(New_X_mb, device="cuda")
            else:
                X_mb = torch.tensor(X_mb)
                M_mb = torch.tensor(M_mb)
                H_mb = torch.tensor(H_mb)
                New_X_mb = torch.tensor(New_X_mb)

            optimizer_D.zero_grad()
            D_loss_curr = self.discriminator_loss(
                M=M_mb, New_X=New_X_mb, H=H_mb)
            D_loss_curr.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            G_loss_curr, MSE_train_loss_curr, MSE_test_loss_curr = self.generator_loss(
                X=X_mb, M=M_mb, New_X=New_X_mb, H=H_mb)
            G_loss_curr.backward()
            optimizer_G.step()

            # %% Intermediate Losses
            if it % 100 == 0:
                print('Iter: {}'.format(it), end='\t')
                print('Train_loss: {:.4}'.format(
                    np.sqrt(MSE_train_loss_curr.item())))
                #print('Test_loss: {:.4}'.format(np.sqrt(MSE_test_loss_curr.item())))

    # reconstruct data

    def _transform(self):

        def split_list(l, n):
            for idx in range(0, len(l), n):
                yield l[idx:idx + n]

        id_list_list = split_list(list(range(self.No)), self.mb_size)

        reconst_array = False

        for i in id_list_list:
            mb_idx = i
            diff = len(mb_idx)-self.mb_size
            if diff < 0:
                for j in range(-diff):
                    mb_idx.append(0)

            X_mb = self.Data[mb_idx, :]
            M_mb = self.Missing[mb_idx, :]
            Z_mb = sample_Z(self.mb_size, self.Dim)
            New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb  # Missing Data Introduce

            if self.use_gpu is True:
                X_mb = torch.tensor(X_mb, device='cuda')
                M_mb = torch.tensor(M_mb, device='cuda')
                New_X_mb = torch.tensor(New_X_mb, device='cuda')
            else:
                X_mb = torch.tensor(X_mb)
                M_mb = torch.tensor(M_mb)
                New_X_mb = torch.tensor(New_X_mb)

            MSE_final, Sample = self.test_loss(X=X_mb, M=M_mb, New_X=New_X_mb)
            imputed_data = M_mb * X_mb + (1-M_mb) * Sample

            imp = imputed_data.cpu().detach().numpy()

            if type(reconst_array) != type(False):
                reconst_array = np.concatenate([reconst_array, imp])
            else:
                reconst_array = imp

        return reconst_array[:self.No]


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return np.random.normal(size=size, scale=xavier_stddev)

# Hint Vector Generation


def sample_M(m, n, p):
    A = np.random.uniform(0., 1., size=[m, n])
    B = A > p
    C = 1.*B
    return C

# %% 3. Other functions
# Random sample generator for Z


def sample_Z(m, n):
    return np.random.uniform(0., 0.01, size=[m, n])

# Mini-batch generation


def sample_idx(m, n):
    A = np.random.permutation(m)
    idx = A[:n]
    return idx
