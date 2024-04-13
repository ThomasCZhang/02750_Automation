import numpy as np
from sklearn.utils import resample
from sklearn.base import clone
from sklearn.metrics import log_loss

from UpdaterClass import *
import numpy as np

class IWAL(ActiveLearningUpdater):
    def __init__(self, x_lab, y_lab, x_unlab, y_unlab,
                unlab_cost: np.ndarray=None, aggressive: bool=True, seed: int=0, pmin: float = 0.1, b:int=None) -> None:
        super().__init__(x_lab, y_lab, x_unlab,y_unlab,unlab_cost, aggressive, seed)
        self.pmin = pmin
        self.weights = np.ones(self.x_lab.shape[0])*pmin
        if b is not None:
            self.b = b
        else: 
            self.b =(x_lab.shape[0] + x_unlab.shape[0])//5
    
    def Update(self, clf,  batch_size:int=1, n_models:int=50):
        """
        Updates the labeled and unlabeled data by performing one iteration of IWAL.
        Input:
        clf: The classifier
        batch_size (int): Number of samples to move from unlabeled to labeled set.
        n_models: The number of models to use for creating a committee to calculate IWAL threshold.
        """
        utility_vector = self.CalculateUtility()
        
        if self.aggressive:
            next_indicies = np.argsort(utility_vector)[-batch_size:]
        else:
            utility_vector = utility_vector/np.sum(utility_vector)
            next_indicies = self.rng.choice(np.arange(utility_vector.shape[0]), size=batch_size, p=utility_vector)

        for next_idx in reversed(sorted(next_indicies)):
            pt = self.IWALRejectionThreshold(clf, next_idx, n_models)
            rn = self.rng.uniform(0., 1., 1)
            if rn < pt:
                self.UpdateLabeledData(next_idx)
                self.weights = np.append(self.weights, [self.pmin/pt])
            else:
                self.UpdateLabeledData(next_idx, False)


    def CalculateUtility(self):
        utility = self.rng.random(self.x_unlab.shape[0])
        return utility
    
    def IWALRejectionThreshold(self, clf, next_idx:int, n_models: int=50):
        """
        Performs IWAL sampling.
        Input:
        clf: The classifier.
        next_idx: The index of the unlabeled sample to calculate a threshold for.
        b (int): The minimum size before begining bootstrapping.
        k (int): the number of boostrapped distributions to make.

        Output:
        (float) pt, the probability of taking new_x
        """
        new_x = self.x_unlab[next_idx]
        labels = np.unique(self.y_lab)
        n_classes = len(labels)
        new_x = new_x.reshape(1, -1)
        if self.x_lab.shape[0] <= self.b or n_classes == 1: # We need to have at least 2 classes to fit.
            pt = 1
        elif self.x_lab.shape[0] > self.b:
            # Create a committee
            bootstraped_train_data = [resample(self.x_lab, self.y_lab, replace = True,  stratify = self.y_lab) for _ in range(n_models)]
            bs_train_x = [tmp[0] for tmp in bootstraped_train_data]
            bs_train_y = [tmp[1] for tmp in bootstraped_train_data]

            models = [clone(clf) for _ in range(n_models)]
            for j in range(n_models):
                models[j].fit(bs_train_x[j], bs_train_y[j], sample_weight = self.weights)

            # Find the maximum loss difference between the models.
            losses = []
            for label in labels:
                losses.append(
                np.array([log_loss(np.array(label).reshape(1), model.predict_proba(new_x), labels = labels) for model in models])
                )
            max_loss_diff = np.max([np.max(arr)-np.min(arr) for arr in losses])

            pt = self.pmin + (1-self.pmin)*max_loss_diff

        return pt

