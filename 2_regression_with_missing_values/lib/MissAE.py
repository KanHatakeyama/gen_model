"""
Imputer class of autoencoder or variational autoencoder

"""
import torchvision.transforms as transforms
import torch
from torch import optim
import torch.nn as nn
import numpy as np
import copy
from VAE import VAE, AE, device
from EarlyStopping import EarlyStopping


MASK_VALUE = -5


class MissAE:
    def __init__(self, mode="AE", z_dim=10, batch_norm=False, patience=5):
        """
        mode: "AE" or "VAE"
        z_dim: dimension of z space
        batch_norm: batch normalization for the output (not recommended)
        patience: patience for early stopping
        """
        self.mode = mode
        self.z_dim = z_dim
        self.batch_norm = batch_norm
        self.patience = patience

    # impute
    def fit_transform(self, dat_list, verbose=0, max_epoch=30):
        """
        dat_list: list (or numpy array) of data to be imputed
        max_epoch: max number of imputation
        return: numpy array of imputed data
        """
        model = self.model

        if self.mode == "AE":
            self.model = AE(z_dim=self.z_dim,
                            input_dim=msk.shape[1], batch_norm=self.batch_norm)
        elif self.mode == "VAE":
            self.model = VAE(z_dim=self.z_dim, input_dim=msk.shape[1])
        else:
            print("err: select VAE or AE")

        # preprocess
        msk = np.array(dat_list)
        msk[np.where(msk != msk)] = MASK_VALUE
        dat_list = np.array(dat_list)
        dat_list = np.nan_to_num(dat_list)
        replaced_data_list = copy.copy(dat_list)

        self.diff_list = []
        es = EarlyStopping(patience=self.patience)

        # imputation loop
        for i in (range(max_epoch)):
            model = prepare_model(replaced_data_list, model, verbose=verbose)
            replaced_data_list, diff = fill_data(
                replaced_data_list, msk, model, plot=False)

            print("epoch {} reconstruction diff {}".format(i, diff))
            self.diff_list.append(diff)

            if es.validate(diff):
                break
        self.model = model

        return np.array(replaced_data_list)


def prepare_loader(dat_list, shuffle=True, batch_size=100):
    dat_list = np.array(dat_list)
    dat_list = dat_list.reshape(dat_list.shape[0], dat_list.shape[1], -1)

    dataset_train = Dataset(np.array(dat_list), dat_list, transform)

    trainloader = torch.utils.data.DataLoader(dataset_train,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=4,
                                              drop_last=False)
    return trainloader


# prepare trained model
def prepare_model(dat_list, model, verbose=0, epoch=20, lr=0.001):

    trainloader = prepare_loader(dat_list)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    nan_count = 0
    prev_model = None
    for i in range(epoch):
        losses = []
        for x, _ in trainloader:
            x = x.to(device, dtype=torch.float)
            model.zero_grad()
            y = model(x)
            loss = model.loss(x)

            # avoid nan
            if torch.isnan(loss):
                if prev_model:
                    model = prev_model
                nan_count += 1
                if nan_count > 5:
                    return model
            else:
                prev_model = copy.deepcopy(model)
                loss.backward()
                optimizer.step()

            losses.append(loss.cpu().detach().numpy())
        if verbose:
            print("mini_epoch: {} loss: {}".format(i, np.average(losses)))
    return model


# impute data with a trained model
def fill_data(dat_list, mask_list, model, plot=True, loss=nn.MSELoss()):

    reconst_loader = prepare_loader(dat_list, shuffle=False, batch_size=32)
    mask_loader = prepare_loader(mask_list, shuffle=False, batch_size=32)

    model.eval()
    replaced_data_list = []

    flg = True
    diff = 0
    for a, b in (zip(reconst_loader, mask_loader)):
        x = a[0]
        mask = b[0]

        x = x.to(device, dtype=torch.float)

        # generate from x
        y, _ = model(x)

        # make new data with mask & reconstruction data
        mask = mask.to(device, dtype=torch.float)

        # only masking part will be left
        y[torch.where(mask != MASK_VALUE)] = 0

        # missing part of the original data will be zero
        mask[torch.where(mask == MASK_VALUE)] = 0

        # final form
        filled_data = mask+y

        # calc difference
        diff += loss(x, filled_data).cpu().detach().numpy()

        replaced_data_list.extend(filled_data.view(
            filled_data.shape[0], -1).cpu().detach().numpy())

    return replaced_data_list, diff


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])


class Dataset(torch.utils.data.Dataset):

    def __init__(self, data, label, transform=None):
        self.transform = transform
        self.data = data
        self.label = label
        self.data_num = len(data)

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        if self.transform:
            out_data = self.transform(self.data[idx])
        else:
            out_data = self.data[idx]

        return out_data, self.label[idx]
