import numpy as np
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.base import clone
from sklearn.metrics import log_loss, hinge_loss

def IWALRejectionThreshold(clf, new_x: np.ndarray, train_x: np.ndarray, train_y: np.ndarray, weights: np.ndarray,
                  b: int, k: int, pmin: float):
    """
    Performs IWAL sampling.
    Input:
    new_x (np.ndarray): The sample that we are calculating pt for
    train_x (np.ndarray): The data that is currently in the training set.
    train_y (np.ndarray): The labels of the data in the training set.
    weights (np.ndarray): the weights of the samples in the training set.
    b (int): The minimum size before begining bootstrapping.
    k (int): the number of boostrapped distributions to make.
    pmin (int): The minimum sampling probability

    Output:
    (float) pt, the probability of taking new_x
    """
    labels = np.unique(train_y)
    n_classes = len(labels)
    new_x = new_x.reshape(1, -1)
    if train_x.shape[0] <= b or n_classes == 1: # We need to have at least 2 classes to fit.
        pt = 1
    elif train_x.shape[0] > b:
        # Create a committee
        bootstraped_train_data = [resample(train_x, train_y, replace = True,  stratify = train_y) for _ in range(k)]
        bs_train_x = [tmp[0] for tmp in bootstraped_train_data]
        bs_train_y = [tmp[1] for tmp in bootstraped_train_data]

        models = [clone(clf) for _ in range(k)]
        for j in range(k):
            models[j].fit(bs_train_x[j], bs_train_y[j], sample_weight = weights)

        # Find the maximum loss difference between the models.
        losses = []
        for label in labels:
            losses.append(
            np.array([log_loss(np.array(label).reshape(1), model.predict_proba(new_x), labels = labels) for model in models])
            )
        max_loss_diff = np.max([np.max(arr)-np.min(arr) for arr in losses])

        pt = pmin + (1-pmin)*max_loss_diff

    return pt

def IWAL(clf, new_x: np.ndarray, train_x: np.ndarray, train_y: np.ndarray, weights: np.ndarray,
                  b: int, k: int, pmin: float, rng: np.random.Generator):
    """
    Performs IWAL sampling.
    Input:
    new_x (np.ndarray): The sample that we are calculating pt for
    train_x (np.ndarray): The data that is currently in the training set.
    train_y (np.ndarray): The labels of the data in the training set.
    weights (np.ndarray): the weights of the samples in the training set.
    b (int): The minimum size before begining bootstrapping.
    k (int): the number of boostrapped distributions to make.
    pmin (int): The minimum sampling probability
    rng (np.random.Generator): A seeded numpy random number generator

    Output:
    (float) pt, the probability of taking new_x or None if new_x is rejected.
    """
    pt = IWALRejectionThreshold(clf, new_x, train_x, train_y, weights, b, k, pmin)
    rn = rng.uniform(0., 1., 1)
    if rn < pt:
        return pt
    return None
