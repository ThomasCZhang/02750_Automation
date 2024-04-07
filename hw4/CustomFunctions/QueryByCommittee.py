from CustomFunctions.UpdaterClass import *
import numpy as np

class QueryByCommittee(ActiveLearningUpdater):
    def __init__(self, x_lab, y_lab, x_unlab, y_unlab, aggressive: bool=True, seed: int=0) -> None:
        super().__init__(x_lab, y_lab, x_unlab,y_unlab, aggressive, seed)
        pass

    def Update(self, clf, n_models: int = 50):
        """
        Updates the labeled and unlabeled data by performing one iteration of active learning.
        Input:
        clf: The classifier
        n_models: the number of models to use to generate the committee
        """
        utility_vector = self.CalculateUtility(clf, n_models)
        
        if self.aggressive:
            next_idx = np.argmax(utility_vector)
        else:
            utility_vector = utility_vector/np.sum(utility_vector)
            next_idx = self.rng.choice(np.arange(utility_vector.shape[0]), p=utility_vector)

        self.UpdateLabeledData(next_idx)

    def CalculateUtility(self, clf, n_models: int=50):
        """
        Returns the utility vector of unlabeled data used by query by committee.
        Input:
        clf: the classifier
        n_models: Number of models in the committee

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

        return disagreement

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