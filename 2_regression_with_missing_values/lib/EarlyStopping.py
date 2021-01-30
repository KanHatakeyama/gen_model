import numpy as np


class EarlyStopping():
    def __init__(self, patience=3, verbose=1):
        self._step = 0
        self._loss = float('inf')
        self.patience = patience
        self.verbose = verbose

    def validate(self, loss):
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    print('early stopping')
                return True
        else:
            self._step = 0
            self._loss = loss

        return False


def check_converge(log, duration=20, threshold=0.01):
    ave = np.mean(log[-duration:])
    current = log[-1]

    if abs(ave-current) < threshold and len(log) > duration:
        print("early stopping")
        return True
    else:
        return False
