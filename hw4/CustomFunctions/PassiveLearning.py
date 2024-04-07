from CustomFunctions.UpdaterClass import *
import numpy as np

class PassiveLearning(ActiveLearningUpdater):
    def __init__(self, x_lab, y_lab, x_unlab, y_unlab, aggressive: bool=True, seed: int=0) -> None:
        super().__init__(x_lab, y_lab, x_unlab,y_unlab, aggressive, seed)
        pass
    
    def Update(self, clf):
        """
        Updates the labeled and unlabeled data by performing one iteration of active learning.
        Input:
        clf: The classifier
        """
        utility_vector = self.CalculateUtility(clf)
        
        if self.aggressive:
            next_idx = np.argmax(utility_vector)
        else:
            utility_vector = utility_vector/np.sum(utility_vector)
            next_idx = self.rng.choice(np.arange(utility_vector.shape[0]), p=utility_vector)

        self.UpdateLabeledData(next_idx)

    def CalculateUtility(self):
        utility = np.zeros(self.x_unlab.shape[0])
        idx = self.rng.integers(0, self.x_unlab.shape[0])
        utility[idx] = 1
        return utility

