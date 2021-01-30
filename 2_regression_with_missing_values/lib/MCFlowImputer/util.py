import os
import sys
import csv
import wget
import zipfile
import numpy as np
import pandas as pd
import torch
from torch import nn
import torchvision
from torchvision import transforms, datasets
from torch import distributions
import ssl
from sklearn.metrics import mean_squared_error


'''
Helper functions for MCFlow
'''

# normalize input (0-1)


def preprocess(data):
    maxs = np.zeros(data.shape[1])
    mins = np.zeros(data.shape[1])
    for idx, row in enumerate(data.T):
        maxs[idx] = row.max()
        mins[idx] = row.min()
    dat = []
    for idx, row in enumerate(data):
        rw = []
        for idx_v, value in enumerate(row):
            rw.append((value - mins[idx_v]) / (maxs[idx_v] - mins[idx_v]))
        dat.append(np.asarray(rw))

    return np.asarray(dat), maxs, mins


# imputation
def endtoend_train(flow, nn_model, nf_optimizer, nn_optimizer, loader, use_cuda, skip_nn=False, loss_ratio=1.0):

    nf_totalloss = 0
    totalloss = 0
    total_log_loss = 0
    total_imputing = 0
    loss_func = nn.MSELoss(reduction='none')

    for index, (vectors, labels) in enumerate(loader):
        if use_cuda:
            vectors = vectors.cuda()
            labels[0] = labels[0].cuda()
            labels[1] = labels[1].cuda()

        z, nf_loss = flow.log_prob(vectors, use_cuda)
        nf_totalloss += nf_loss.item()

        if not skip_nn:
            z_hat = nn_model(z)
        else:
            z_hat = z
        x_hat = flow.inverse(z_hat)
        _, log_p = flow.log_prob(x_hat, use_cuda)

        batch_loss = torch.sum(loss_func(x_hat, labels[0]) * (1 - labels[1]))
        batch_loss = batch_loss*loss_ratio
        total_imputing += np.sum(1-labels[1].cpu().numpy())

        log_lss = log_p
        total_log_loss += log_p.item()
        totalloss += batch_loss.item()
        batch_loss += log_lss
        nf_loss.backward(retain_graph=True)
        nf_optimizer.step()
        nf_optimizer.zero_grad()
        batch_loss.backward()
        nn_optimizer.step()
        nn_optimizer.zero_grad()

    index += 1
    return totalloss, total_log_loss/index, nf_totalloss/index


def create_mask(shape):
    zeros = int(shape/2)
    ones = shape - zeros
    lst = []
    for i in range(shape):
        if zeros > 0 and ones > 0:
            if np.random.uniform() > .5:
                lst.append(0)
                zeros -= 1
            else:
                lst.append(1)
                ones -= 1
        elif zeros > 0:
            lst.append(0)
            zeros -= 1
        else:
            lst.append(1)
            ones -= 1
    return np.asarray(lst)


def init_flow_model(num_neurons, num_layers, init_flow, data_shape, use_cuda):

    def nets(): return nn.Sequential(nn.Linear(data_shape, num_neurons), nn.LeakyReLU(), nn.Linear(num_neurons, num_neurons), nn.LeakyReLU(), nn.Linear(num_neurons, num_neurons),
                                     nn.LeakyReLU(), nn.Linear(num_neurons, data_shape), nn.Tanh())

    def nett(): return nn.Sequential(nn.Linear(data_shape, num_neurons), nn.LeakyReLU(), nn.Linear(num_neurons, num_neurons), nn.LeakyReLU(),
                                     nn.Linear(num_neurons, num_neurons),  nn.LeakyReLU(), nn.Linear(num_neurons, data_shape))

    mask = []
    for idx in range(num_layers):
        msk = create_mask(data_shape)
        mask.append(msk)
        mask.append(1-msk)

    masks = torch.from_numpy(np.asarray(mask)).float()
    if use_cuda:
        masks = masks.cuda()
    prior = distributions.MultivariateNormal(
        torch.zeros(data_shape), torch.eye(data_shape))
    flow = init_flow(nets, nett, masks, prior)
    if use_cuda:
        flow.cuda()

    return flow


def inference_imputation_networks(nn, nf, data, use_cuda):
    lst = []

    batch_sz = 8
    iterations = int(data.shape[0]/batch_sz)
    left_over = data.shape[0] - batch_sz * iterations

    with torch.no_grad():
        for idx in range(iterations):
            rows = data[int(idx*batch_sz):int((idx+1)*batch_sz)]
            if use_cuda:
                rows = torch.from_numpy(rows).float().cuda()
            else:
                rows = torch.from_numpy(rows).float()

            z = nf(rows)[0]
            z_hat = nn(z)
            x_hat = nf.inverse(z_hat)

            lst.append(np.clip(x_hat.cpu().numpy(), 0, 1))

        rows = data[int((idx+1)*batch_sz):]
        if use_cuda:
            rows = torch.from_numpy(rows).float().cuda()
        else:
            rows = torch.from_numpy(rows).float()

        z = nf(rows)[0]
        z_hat = nn(z)
        x_hat = nf.inverse(z_hat)

        lst.append(np.clip(x_hat.cpu().numpy(), 0, 1))

    final_lst = []
    for idx in range(len(lst)):
        for element in lst[idx]:
            final_lst.append(element)

    return final_lst


def calc_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
