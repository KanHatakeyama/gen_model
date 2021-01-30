from tqdm.notebook import tqdm
import numpy as np
import torch
from ImputationDataLoader import ImputationDataLoader
import copy
import util
from util import calc_rmse
from InterpRealNVP import InterpRealNVP
from LatentToLatentApprox import LatentToLatentApprox
import itertools

# A class to impute values by MCFlow


class MCFlowImputer:
    def __init__(self,
                 batch_size=8,
                 num_nf_layers=3,
                 n_epochs=128,
                 lr=10**-4,
                 use_cuda=True,
                 verbose=False,
                 loss_ratio=1.0
                 ):
        """
         batch_size: batch size
         num_nf_layers: number of hidden layers
         n_epochs: number of epochs
         lr: learning rate
         use_cuda=True: use cuda or not
         verbose: verbose
         loss_ratio: ratio of loss for flow model and reconstruction error. if 1>loss_ratio, the model will fit the training data more strongly.        
         return: list of imputed train,test, and validation arrays (numpy)
        """

        self.batch_size = batch_size
        self.num_nf_layers = num_nf_layers
        self.n_epochs = n_epochs
        self.lr = lr
        self.use_cuda = use_cuda
        self.verbose = verbose
        self.loss_ratio = loss_ratio

        if not torch.cuda.is_available() or self.use_cuda == False:
            print("CUDA Unavailable. Using cpu. Check torch.cuda.is_available()")
            self.use_cuda = False

    # fit transform function
    def fit_transform(self,
                      train_array,
                      test_array=None,
                      val_array=None,
                      val_y=None,
                      target_id=None,
                      n_repeat=1,
                      ):
        """
         train_array: numpy array for train dataset
         test_array: numpy array for test dataset
         val_array: validation array (optional, not used in the article)
         val_y: answer (y) for the validation array
         target_id: column ID of y in the validation array
         n_repeat: repeat imputation (i.e., MCFlow is a random process)

        """

        self.best_loss = np.inf
        best_tr_X = None

        # repeat whole process to find best array (i.e., imputation is a random process)
        for i in range(n_repeat):

            # initialize dataset
            ldr = ImputationDataLoader(train_array)

            if test_array is not None:
                ldr_te = ImputationDataLoader(test_array)

            if val_array is not None:
                ldr_val = ImputationDataLoader(val_array)

            data_loader = torch.utils.data.DataLoader(
                ldr, batch_size=self.batch_size, shuffle=True, drop_last=False)
            num_neurons = int(ldr.matrix[0].shape[0])

            # Initialize normalizing flow model neural network and its optimizer
            flow = util.init_flow_model(
                num_neurons, self.num_nf_layers, InterpRealNVP, ldr.matrix[0].shape[0], self.use_cuda)
            nf_optimizer = torch.optim.Adam(
                [p for p in flow.parameters() if p.requires_grad == True], lr=self.lr)

            # Initialize latent space neural network and its optimizer
            num_hidden_neurons = [int(ldr.matrix[0].shape[0]), int(ldr.matrix[0].shape[0]), int(
                ldr.matrix[0].shape[0]), int(ldr.matrix[0].shape[0]),  int(ldr.matrix[0].shape[0])]
            nn_model = LatentToLatentApprox(
                int(ldr.matrix[0].shape[0]), num_hidden_neurons).float()
            if self.use_cuda:
                nn_model.cuda()
            nn_optimizer = torch.optim.Adam(
                [p for p in nn_model.parameters() if p.requires_grad == True], lr=self.lr)

            reset_scheduler = 2

            #train and impute
            for epoch in tqdm(range(self.n_epochs)):
                loss, _, _ = util.endtoend_train(
                    flow, nn_model, nf_optimizer, nn_optimizer, data_loader, self.use_cuda, loss_ratio=self.loss_ratio)

                if self.verbose:
                    print("Epoch", epoch, " Loss", loss)

                if (epoch+1) % reset_scheduler == 0:
                    # impute arrays
                    ldr.reset_imputed_values(nn_model, flow, self.use_cuda)
                    if self.verbose:
                        print(epoch, ": imputed")

                    if test_array is not None:
                        # predict
                        ldr_te.reset_imputed_values(
                            nn_model, flow, self.use_cuda)

                    if val_array is not None:
                        ldr_val.reset_imputed_values(
                            nn_model, flow, self.use_cuda)

                        # validate
                        if target_id is not None:
                            pred_y = ldr_val.matrix[:, target_id]
                            rmse = calc_rmse(val_y, pred_y)
                            print(rmse)

                            if rmse < self.best_loss:
                                self.best_loss = rmse
                                self.best_tr_X = copy.copy(ldr.matrix)
                                self.best_val_X = copy.copy(ldr_val.matrix)
                                if test_array is not None:
                                    self.best_te_X = copy.copy(ldr_te.matrix)

                    if val_array is None:
                        self.best_model_data = [
                            copy.copy(flow), copy.copy(nn_model)]

                    # init model
                    flow = util.init_flow_model(
                        num_neurons, self.num_nf_layers, InterpRealNVP, ldr.matrix[0].shape[0], self.use_cuda)
                    nf_optimizer = torch.optim.Adam(
                        [p for p in flow.parameters() if p.requires_grad == True], lr=self.lr)
                    reset_scheduler = int(reset_scheduler*2)

        res_list = [ldr.matrix]
        if test_array is not None:
            res_list.append(ldr_te.matrix)
        if val_array is not None:
            res_list.append(ldr_val.matrix)

        return res_list

    # calculate z-space of the input (call this function after fit)
    def calc_z(self, array):
        """
        array: numpy array to be imputed
        return: z-space values for the input (numpy)
        """
        ldr = ImputationDataLoader(array)
        data_loader = torch.utils.data.DataLoader(
            ldr, batch_size=32, shuffle=False, drop_last=False)
        flow, nn_model = self.best_model_data[0], self.best_model_data[1]

        prediction_list = []

        with torch.no_grad():
            for index, (vectors, labels) in enumerate(data_loader):
                z, _ = flow.to("cpu").log_prob(vectors, False)
                z_hat = nn_model.to("cpu")(z)
                prediction_list.append(z_hat.detach().cpu().numpy())

        pred_array = np.array(
            list(itertools.chain.from_iterable(prediction_list)))
        return pred_array
