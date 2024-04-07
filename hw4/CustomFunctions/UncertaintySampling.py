from CustomFunctions.UpdaterClass import *
import numpy as np

class UncertaintySampling(ActiveLearningUpdater):
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