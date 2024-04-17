from UpdaterClass import *
import numpy as np
import torch

class DBAL(ActiveLearningUpdater):
    def __init__(self, x_lab, y_lab, x_unlab, y_unlab, unlab_cost, task, aggressive: bool = True, seed: int = 0) -> None:
        super().__init__(x_lab, y_lab, x_unlab, y_unlab, unlab_cost, task, aggressive, seed)

    def Update(self, clf, batch_size:int=1):
        """
        Updates the labeled and unlabeled data by performing one iteration of deep batch active learning.
        Input:
        clf: The classifier
        batch_size (int): Number of samples to move from unlabeled to labeled set.
        """
        next_indices = self.ChooseIndicies(clf, batch_size)
        self.UpdateLabeledData(next_indices)

    def ChooseIndicies(self, clf, batch_size):
        clf.fit(self.x_lab, self.y_lab)
        preds = self.GetEnsamblePreds(clf) # N x C matrix. N = number of samples, C = number of classes.
        # var = np.diag(np.cov(np.mean(preds, axis = 1)))
        var = np.diag(self.Covariance(preds))
        var = var/np.sum(var)

        converged = False
        init_num_batches = 100
        m = 50

        batch_indices = [self.rng.choice(
            np.arange(self.x_unlab.shape[0]), size=batch_size, p=var) for _ in range(init_num_batches)]
        batch_scores = [self.CalculateScore(indices, preds) for indices in batch_indices]
        top_m_batches = np.argsort(batch_scores)[-m:]

        batch_indices = [batch_indices[i] for i in top_m_batches]
        batch_scores = [batch_scores[i] for i in top_m_batches]

        converged = False
        idx = 0
        last_update = idx
        while not converged:
            updated = self.UpdateBatches(idx, preds, batch_indices, batch_scores)
            if updated:
                last_update = idx
            idx = (idx + 1)%batch_size
            if last_update == idx:
                converged = True
    
        return batch_indices[np.argmax(batch_scores)]

    def CalculateScore(self, indices, preds):
        # cov = np.cov(np.cov(np.mean(preds, axis = 1)))
        cov = self.Covariance(preds[indices])
        score = np.log(np.linalg.det(cov + 1e-6*np.eye(cov.shape[0])))
        if np.isnan(score):
            print(cov, np.linalg.det(cov))
            raise Exception()
        return score

    def UpdateBatches(self, idx, preds, batch_indices, batch_scores) -> bool:
        converged = True
        for i, indices in enumerate(batch_indices):
            best_score = batch_scores[i]
            best_indices = indices.copy()

            non_batch_indices = np.delete(np.arange(self.x_unlab.shape[0]), indices)
            for j in non_batch_indices:
                indices[idx] = j
                score = self.CalculateScore(indices, preds)
                if score > best_score:
                    converged = False
                    best_score = score
                    best_indices = indices.copy()
            
            batch_indices[i] = best_indices
            batch_scores[i] = best_score
        return converged

    def GetEnsamblePreds(self, clf, n_passes:int = 50):
        clf.train()
        with torch.no_grad():
            preds = [clf(self.x_unlab) for _ in range(n_passes)]
        preds = np.array(preds)
        preds = np.swapaxes(preds, 0, 1)
        # yhat = np.mean(yhat, axis = 0)
        return preds
    
    def Covariance(self, preds):
        if preds.ndim == 2:
            return np.cov(preds)
        
        batch_size, num_passes, num_classes = preds.shape # num_passes is the number of forward passes with dropout
        cov = (preds - np.mean(preds, axis = 1, keepdims=True)).reshape(batch_size, -1)
        # cov = (cov[None, ...] * cov[:, None, ...]).sum(axis=-1) / num_passes
        cov = cov@cov.T/num_passes
        return cov
    
