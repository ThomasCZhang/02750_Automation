from CustomFunctions.UpdaterClass import *
import numpy as np

class UncertaintySampling(ActiveLearningUpdater):
    def __init__(self, x_lab, y_lab, x_unlab, y_unlab, aggressive: bool=True, seed: int=0) -> None:
        super().__init__(x_lab, y_lab, x_unlab,y_unlab, aggressive, seed)
        pass
    
    def Update(self, clf, batch_size: int=1):
        """
        Updates the labeled and unlabeled data by performing one iteration of active learning.
        Input:
        clf: The classifier
        batch_size (int): Number of samples to move from unlabeled to labeled set.
        """
        utility_vector = self.CalculateUtility(clf)
        
        if self.aggressive:
            next_indicies = np.argsort(utility_vector)[-batch_size:]
        else:
            utility_vector = utility_vector/np.sum(utility_vector)
            next_indicies = self.rng.choice(np.arange(utility_vector.shape[0]), size=batch_size, p=utility_vector)

        self.UpdateLabeledData(next_indicies)

    def CalculateUtility(self, clf):
        """
        Returns the utility vector (uncertainty) of the unlabeled data used by Uncertainty sampling. 
        Input:
        clf: the classifier

        Output:
        (np.ndarray) The utility vector that should be maximized to find the next index
        """
        active_model = clf.fit(self.x_lab, self.y_lab)
        try:
            uncertainty = active_model.predict_proba(self.x_unlab)
        except:
            uncertainty = np.sum(self.x_unlab@np.linalg.pinv(self.x_unlab.T@self.x_unlab)*self.x_unlab, axis=1)
            uncertainty = uncertainty[:, np.newaxis]
        entropy = uncertainty*np.log2(uncertainty+1e-20)
        entropy = -np.sum(entropy, axis = 1)
        return entropy