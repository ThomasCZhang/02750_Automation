from UpdaterClass import *
import numpy as np

class DensitySampling(ActiveLearningUpdater):
    def __init__(self, x_lab, y_lab, x_unlab, y_unlab,
                unlab_cost: np.ndarray=None, aggressive: bool=True, seed: int=0) -> None:
        super().__init__(x_lab, y_lab, x_unlab,y_unlab, unlab_cost, aggressive, seed)
        self.densities = self.CalculateDensity()
        pass

    def Update(self, clf, batch_size:int=1, n_models:int=50, beta:float=0.7):
        """
        Updates the labeled and unlabeled data by performing one iteration of active learning.
        Input:
        clf: The classifier
        batch_size (int): Number of samples to move from unlabeled to labeled set.
        n_models: Number of models to use for query by committee.
        """
        utility_vector = self.CalculateUtility(clf,n_models,beta)
        
        if self.aggressive:
            next_indicies = np.argsort(utility_vector)[-batch_size:]
        else:
            utility_vector = utility_vector/np.sum(utility_vector)
            next_indicies = self.rng.choice(np.arange(utility_vector.shape[0]), size=batch_size, p=utility_vector)
       
        for next_idx in reversed(sorted(next_indicies)):
            self.UpdateDensity(next_idx)
            self.UpdateLabeledData(next_idx)
    
    def CalculateUtility(self, clf,n_models:int=50,beta:float=0.7):
        """
        Returns the utility vector of unlabeled data used by Density based sampling.
        Input:
        clf: the classifier
        n_models: the number of models to use for query by commitee

        Output:
        (np.ndarray) The utility vector that should be maximized to find the next index
        """
        n_classes = len(np.unique(self.y_lab))
        bootstraped_train_data = [resample(self.x_lab, self.y_lab, replace = True,  stratify = self.y_lab) for _ in range(n_models)]
        bs_train_x = [tmp[0] for tmp in bootstraped_train_data]
        bs_train_y = [tmp[1] for tmp in bootstraped_train_data]

        models = [clone(clf) for _ in range(n_models)]
        for j in range(n_models):
            models[j].fit(bs_train_x[j], bs_train_y[j])
        prob_ys = np.hstack([models[j].predict_proba(self.x_unlab) for j in range(n_models)])
        prob_ys = prob_ys.reshape(self.x_unlab.shape[0], n_models, n_classes)
        disagreement = self.QBC_Disagreement(prob_ys)
        weighted_disagreement = self.densities*(disagreement)**beta
        return weighted_disagreement

    def CalculateDensity(self):
        """
        Calculates the density of the points in the test data. 

        Output:
        (np.ndarray)  The density of each sample in self.x_unlab
        """
        n_sample = self.x_unlab.shape[0]
        res = np.array([np.sum(np.linalg.norm(self.x_unlab - self.x_unlab[i], axis = 1)) for i in range(n_sample)]) / n_sample
        return res
    
    def UpdateDensity(self, index: int):
        """
        If densities already exists then updates
        the densities to what they would be if data corresponding from the given index was not in the
        test data.
        Input:
        index (int): the index of test_x to remove

        Output:
        (np.ndarray) The density of each sample in test_x excluding the sample that was removed.
        """
        n_sample = self.x_unlab.shape[0]
        if n_sample == 1:
            self.densities = np.array([])
        else:
            dist_matrix = np.linalg.norm(self.x_unlab - self.x_unlab[index]) # Distance from the indexed sample to the all test_x samples.
            self.densities = (self.densities*n_sample - dist_matrix)/(n_sample-1)
            self.densities = np.delete(self.densities, index, axis = 0)

    def QBC_Disagreement(self, pred_ys, type = 'KL'):
        """
        Calculates the disagreement for querry by committee. 

        Input:
        pred_ys (np.ndarray): A tensor of the shape (n_samples, n_predictions, n_classes)

        Return:
        The disagreement for each sample as a (n_sample,) length array.
        """
        if type == 'KL':
            Pm = np.mean(pred_ys, axis = 1, keepdims= True)
            KL = pred_ys*np.log2(pred_ys/Pm +1e-20) # KL Divergence for each sample, model, label pair
            KL = np.sum(KL, axis=(1,2)) # Average KL divergence.
        return KL