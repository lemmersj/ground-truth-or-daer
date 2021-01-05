"""An early stopping and checkpointing script.

Taken from github.com/Bjarten/early-stopping-pytorch"""
import os
import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve
    after a given patience. Saves checkpoints in the meantime.
    """
    def __init__(self, out_dir, patience=7, verbose=False, delta=0):
        """Initializes the early stopping module.

        Args:
            out_dir: The folder in which to save the best weight checkpoint
            patience (int): How long to wait after last time validation
            loss improved. Default: 7

            verbose (bool): If True, prints a message for each validation loss
            improvement. Default: False

            delta (float): Minimum change in the monitored quantity to qualify
            as an improvement. Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.out_dir = out_dir

    def __call__(self, val_loss, model):
        """Updates the early stopping tracker.

        Accepts a validation loss, compares it to the previous best
        loss, saves if best and exits if no improvement in the prescribed
        number of epochs.

        Args:
            val_loss: The current epoch's validation loss.
            model: The current model

        Returns:
            None
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            #print(
            #    'EarlyStopping: {} out of {}'.format(
            #        (self.counter,self.patience)))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases.

            Args:
                val_loss: The current epoch's validation loss (used only
                in verbose mode).
                model: The current model to save
            Returns:
                None
        """

        #if self.verbose:
        #    print("Validation loss decreased ({} --> {}).\
        #          Saving model ...".format((self.val_loss_min, val_loss)))
        torch.save(
            model.state_dict(), os.path.join(
                self.out_dir, 'checkpoint_best_loss.pt'))
        self.val_loss_min = val_loss
