import numpy as np
from sklearn.base import clone
from sklearn.utils import resample

class ActiveLearningUpdater():
    def __init__(self, x_lab, y_lab, x_unlab, y_unlab, unlab_cost, aggressive: bool=True, seed: int=0) -> None:
        """
        Inputs:
        x_lab (np.ndarray): features of labeled data 
        y_lab (np.ndarray): labels of labeled data
        x_unlab (np.ndarray): features of unlabeled data
        y_unlab (np.ndarray): labels of unlabeled data
        aggressive (bool): Use aggressive active learning. Default = True
        """
        self.x_lab = x_lab
        self.y_lab = y_lab
        self.x_unlab = x_unlab
        self.y_unlab = y_unlab
        self.cost = unlab_cost
        self.aggressive = aggressive
        self.rng = np.random.default_rng(seed)
        pass

    def UpdateLabeledData(self, idxs, move:bool=True):
        """
        Updates the labeled data by moving one sample from the unlabeled data to
        the labeled data.
        Input:
        idxs (np.ndarray or int): Numpy array of indicies or single index to move from unlabeled to labeled.
        move (bool): Whether to move the sample from the unlabeled to labeled.
            If false, only deletes sample from unlabeled and does NOT add to labeled set.
        """
        if move:
            self.x_lab = np.vstack((self.x_lab, self.x_unlab[idxs]))
            self.y_lab = np.append(self.y_lab, self.y_unlab[idxs])
        self.x_unlab = np.delete(self.x_unlab, idxs, axis = 0)
        self.y_unlab = np.delete(self.y_unlab, idxs, axis = 0)

    def GetData(self):
        """
        Returns the current labeled and unlabeled data.

        Output: x_labeled, y_labeled, x_unlabeled, y_unlabeled
        """
        return self.x_lab, self.y_lab, self.x_unlab, self.y_unlab




