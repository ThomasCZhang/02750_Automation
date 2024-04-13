from UpdaterClass import *
import numpy as np

class PassiveLearning(ActiveLearningUpdater):
    def __init__(self, x_lab, y_lab, x_unlab, y_unlab,unlab_cost: np.ndarray=None,
                  aggressive: bool=True, seed: int=0) -> None:
        super().__init__(x_lab, y_lab, x_unlab,y_unlab, unlab_cost, aggressive, seed)
        pass
    
    def Update(self, clf, batch_size:int=1):
        """
        Updates the labeled and unlabeled data by performing one iteration of active learning.
        Input:
        clf: The classifier
        batch_size (int): Number of samples to move from unlabeled to labeled set.
        """
        utility_vector = self.CalculateUtility()
        
        if self.aggressive:
            next_indicies = np.argsort(utility_vector)[-batch_size:]
        else:
            utility_vector = utility_vector/np.sum(utility_vector)
            next_indicies = self.rng.choice(np.arange(utility_vector.shape[0]), size=batch_size, p=utility_vector)

        self.UpdateLabeledData(next_indicies)

    def CalculateUtility(self):
        utility = self.rng.random(self.x_unlab.shape[0])
        return utility

