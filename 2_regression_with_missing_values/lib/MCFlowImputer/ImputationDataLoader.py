import numpy as np
import torch
from torch import nn
import sys
import util
import copy

# data loader


class ImputationDataLoader(nn.Module):

    def __init__(self, matrix):
        self.matrix = copy.copy(matrix)
        self.mask = np.isnan(self.matrix).astype(float)
        self.matrix = np.nan_to_num(self.matrix, nan=np.nanmean(matrix))
        _, self.maxs, self.mins = util.preprocess(self.matrix)
        self.original_matrix = copy.copy(self.matrix)

        trans = np.transpose(self.matrix)
        for r_idx, rows in enumerate(trans):
            row = []
            for c_idx, element in enumerate(rows):
                if self.mask[c_idx][r_idx] == 0:
                    row.append(element)
            self.unique_values.append(np.asarray(row))

    def reset_imputed_values(self, nn_model, nf_model,  use_cuda):
        random_mat = np.clip(util.inference_imputation_networks(
            nn_model, nf_model, self.matrix, use_cuda), 0, 1)
        self.matrix = (1-self.mask) * self.original_matrix + \
            self.mask * random_mat

    def __len__(self):
        return len(self.matrix)

    def __getitem__(self, idx):
        return torch.Tensor(self.matrix[idx]), (torch.Tensor(self.original_matrix[idx]), torch.Tensor(self.mask[idx]))
