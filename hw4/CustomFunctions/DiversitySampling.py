from UpdaterClass import *
from sklearn.cluster import KMeans
import numpy as np

class DiversitySampling(ActiveLearningUpdater):
    def __init__(self, x_lab, y_lab, x_unlab, y_unlab,
                unlab_cost: np.ndarray=None,aggressive: bool=True, seed: int=0) -> None:
        super().__init__(x_lab, y_lab, x_unlab,y_unlab,unlab_cost, aggressive, seed)
        pass
    
    def Update(self, clf, batch_size: int=1):
        """
        Updates the labeled and unlabeled data by performing one iteration of diversity sampling.
        Input:
        clf: The classifier
        batch_size (int): Number of samples to move from unlabeled to labeled set.
        """
        utility_vector = self.CalculateUtility(batch_size)
        
        if self.aggressive:
            next_indicies = np.argsort(utility_vector)[-batch_size:]
        else:
            utility_vector = utility_vector/np.sum(utility_vector)
            next_indicies = self.rng.choice(np.arange(utility_vector.shape[0]), size=batch_size, p=utility_vector)

        self.UpdateLabeledData(next_indicies)

    def CalculateUtility(self, batch_size):
        """
        Returns the utility vector (uncertainty) of the unlabeled data used by Diversity Sampling. 
        Input:
        clf: the classifier
        batch_size (int): The number of clusters to form (also the number of samples added in one iteration).

        Output:
        (np.ndarray) The utility vector that should be maximized to find the next index
        """
        n_unlab = self.x_unlab.shape[0]
        cluster_model = KMeans(n_clusters=batch_size)
        cluster_model.fit(self.x_unlab)
        groups = cluster_model.predict(self.x_unlab)
        distances = cluster_model.transform(self.x_unlab)

        mask = np.ones(distances.shape, dtype = bool)
        mask[np.arange(n_unlab), groups] = False

        distances[mask] = np.inf
        
        utility = np.zeros(n_unlab)
        utility[np.argmin(distances, axis = 0)] = 1        

        return utility