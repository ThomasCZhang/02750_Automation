from UpdaterClass import *
import numpy as np

class ExpectedRiskMinimization(ActiveLearningUpdater):
    def __init__(self, x_lab, y_lab, x_unlab, y_unlab,
                 unlab_cost: np.ndarray=None, task: str=None, aggressive: bool=True, seed: int=0) -> None:
        super().__init__(x_lab, y_lab, x_unlab,y_unlab,unlab_cost, task, aggressive, seed)
        pass

    def Update(self, clf, batch_size:int=1):
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
        Returns the utility vector of unlabeled data used by expected risk minimization.
        Input:
        clf: the classifier

        Output:
        (np.ndarray) The utility vector that should be maximized to find the next index
        """

        clf = clf.fit(self.x_lab, self.y_lab)
        if self.x_unlab.shape[0] > 1:
            er_scores = -np.array([self.MEROneSample(clf, i) for i in range(self.x_unlab.shape[0])])
            er_scores = er_scores + np.min(er_scores)
            return er_scores
        else:
            return np.array([1])

    def MEROneSample(self, clf, index) -> float:
        """
        Calculates the expected risk if we add one sample from the test set to the training set.
        Input:
        clf (sklearn predictor): The classifier from Sklearn
        index (int): The index of the test sample that should be added to the training data to calculate expected risk.

        Output:
        (float) The expected risk from adding one sample to the training data. 
        """

        unique_labels = np.unique(self.y_lab)
        n_classes = len(unique_labels)
        prob = clf.predict_proba(self.x_unlab[index:index+1])
        
        models = []
        newTrain_x = np.vstack((self.x_lab, self.x_unlab[index, :]))
        for label in unique_labels:
            newTrain_y = np.append(self.y_lab, label)
            models.append(clone(clf).fit(newTrain_x, newTrain_y))

        newTest_x = np.delete(self.x_unlab, index, axis = 0)

        prob_ys = np.hstack([models[i].predict_proba(newTest_x) for i in range(n_classes)])
        prob_ys = prob_ys.reshape(newTest_x.shape[0], n_classes, n_classes)
        
        score = np.sum(prob*np.sum(1 - np.max(prob_ys, axis = 2), axis = 0))
        return score
  
