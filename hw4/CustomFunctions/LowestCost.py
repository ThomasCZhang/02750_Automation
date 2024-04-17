from UpdaterClass import *
import numpy as np

class LowestCost(ActiveLearningUpdater):
    def __init__(self, x_lab, y_lab, x_unlab, y_unlab,
                unlab_cost: np.ndarray=None, task: str=None, aggressive: bool = True, seed: int = 0) -> None:
        super().__init__(x_lab, y_lab, x_unlab, y_unlab, unlab_cost, task, aggressive, seed)
        if unlab_cost is None:
            raise Exception("Did not give cost vector.")
        self.spent = 0

    def Update(self, clf, batch_size:int=1):
        utility_vector = self.CalculateUtility(clf)
        
        if self.aggressive:
            next_indicies = np.argsort(utility_vector)[-batch_size:]
        else:
            utility_vector = utility_vector/np.sum(utility_vector)
            next_indicies = self.rng.choice(np.arange(utility_vector.shape[0]), size=batch_size, p=utility_vector)

        self.UpdateLabeledData(next_indicies)
        self.spent += np.sum(self.cost[next_indicies])
        self.cost = np.delete(self.cost, next_indicies, axis = 0)

    def CalculateUtility(self, clf):
        """
        Returns the utility vector (uncertainty) of the unlabeled data used by Uncertainty sampling with cost. 
        Input:
        clf: the classifier

        Output:
        (np.ndarray) The utility vector that should be maximized to find the next index
        """
        utility = 1/self.cost * np.min(self.cost)
        return utility
