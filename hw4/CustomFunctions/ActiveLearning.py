import numpy as np
import multiprocessing as mp
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_squared_error, accuracy_score, log_loss

from DOE import *
from DensitySampling import *
from ExpectedRiskMinimization import *
from IWAL import *
from PassiveLearning import *
from QueryByCommittee import *
from UncertaintySampling import *

def InitialSplit(x: np.ndarray, y: np.ndarray,
                seed: int,
                stratify_data: np.ndarray=None,
                init_method: str=None,
                init_frac: float=None
                ):
    """
    Simulates active learning using the designated active learning method.
    Input:
    x (np.ndarray): data (features)
    y (np.ndarray): label of the data
    seed (int): The random seed for splitting the data into train/test.
    stratify_data (np.ndarray): The data to use for stratifying when performing train_test_split.    
    init_method (str): if "DOE" uses DOE to initialize. Otherwise does a stratified train-test split.
    init_frac (float): Value between 0 and 1. Proportion of data to use for initial model.
    end_frac (float): value between 0 and 1. Proportion of data to use for final model after active learning.
    """
    if init_method == "DOE":
        _, starting_k = DOE(x, y, int(init_frac*x.shape[0]), seed=seed, cap=50, metric=D_optimal)
        mask = np.ones(x.shape[0])
        mask[starting_k] = False
        mask = np.array(mask, dtype=bool)
        train_x, train_y = x[starting_k], y[starting_k]
        test_x, test_y = x[mask], y[mask]

    else:
        train_x, test_x, train_y, test_y = train_test_split(x,y, train_size = init_frac,
                                                             shuffle=True, stratify = stratify_data, random_state = seed)        
    return train_x, test_x, train_y, test_y


def CrossValLog(clf, train_x, train_y, test_x, test_y, cv_metrics, params: dict[str, np.ndarray]=None):
    """
    Log the results from cross-validating the training set data.
    """
    cv_res = cross_validate(clf, train_x, train_y, params = params, scoring = cv_metrics)
    res = [np.mean(cv_res['test_'+cv_metric]) for cv_metric in cv_metrics]

    return res

def TestLog(clf, train_x, train_y, test_x, test_y, cv_metrics, params: dict[str, np.ndarray]=None):
    """
    Logs the results from analyzing the unlabeled test dataset.
    """
    clf.fit(train_x, train_y)
    pred_y = clf.predict(test_x)
    acc_res = accuracy_score(test_y, pred_y)
    return [acc_res]

def CrossValAndTestLog(clf, train_x, train_y, test_x, test_y, cv_metrics, params: dict[str, np.ndarray]=None):
    cv_res = cross_validate(clf, train_x, train_y, params = params, scoring = cv_metrics)
    res = [np.mean(cv_res['test_'+cv_metric]) for cv_metric in cv_metrics]
    clf.fit(train_x, train_y)
    pred_y = clf.predict(test_x)
    res.append(mean_squared_error(test_y, pred_y))

    return res

def SimulateAL(x: np.ndarray, y: np.ndarray,
                method: str, seed: int, clf: any ,cv_metrics: list[str],
                stratify_data: np.ndarray=None,
                init_method: str=None,
                init_frac: float=None,
                end_frac: float=None,
                logging_func: any=None,
                aggressive: bool = True,
                batch_size: int=1
                ):
    """
    Simulates active learning using the designated active learning method.
    Input:
    x (np.ndarray): data (features)
    y (np.ndarray): label of the data
    method (str): The active learning method to use.
        PL: Passive Learning
        US: Uncertainty sampling
        QBC: Querry by committee
        ERM: Expected Risk Minimization
        DBS: Density Sampling
        IWAL: Importance Weighted Active Learning
    seed (int): The random seed for splitting the data into train/test.    
    clf (SKlearn classifier): The classifier to use
    cv_metrics (list[str]): Input arguments to the scoring argument of sklearns cross validation.
    stratify_data (np.ndarray): The data to use for stratifying when performing train_test_split.
    init_method (str): if "DOE" uses DOE to initialize. Otherwise does a stratified train-test split.
    init_frac (float): Value between 0 and 1. Proportion of data to use for initial model.
    end_frac (float): value between 0 and 1. Proportion of data to use for final model after active learning.
    logging_func (function): used to calculate the stuff to log. 
    aggressive (bool): Use aggressive active learning. Default is True.
    batch_size (int): The number of samples to add at once. Default is 1.

    Output:
    Returns the accuracy and loss logs across the iterations of the active learning simulation
    """

    preprocess = StandardScaler()
    x = preprocess.fit_transform(x)

    np.random.seed(seed)
    train_x, test_x, train_y, test_y = InitialSplit(x, y, seed, stratify_data=stratify_data,
                                    init_method = init_method, init_frac = init_frac)

    strategy_dict = {
        "PL": PassiveLearning,
        "US": UncertaintySampling,
        "QBC": QueryByCommittee,
        "ERM": ExpectedRiskMinimization,
        "DBS": DensitySampling,
        "IWAL": IWAL,
    }
    active_learning_method = strategy_dict[method](train_x, train_y, test_x, test_y, aggressive)

    size_log = []
    logs = []
    n_iters = int(np.ceil((int(end_frac*x.shape[0])-train_x.shape[0])/batch_size))
    for _ in range(n_iters):
        param = {'sample_weight': active_learning_method.weights} if method == 'IWAL' else None
        train_x, train_y, test_x, test_y = active_learning_method.GetData()

        res = logging_func(clf, train_x, train_y, test_x, test_y, cv_metrics, param)
        logs.append(res)
        size_log.append(train_x.shape[0])

        active_learning_method.Update(clf, batch_size=batch_size)

    param = {'sample_weight': active_learning_method.weights} if method == 'IWAL' else None
    train_x, train_y, test_x, test_y = active_learning_method.GetData()

    np.random.seed(seed)
    res = logging_func(clf, train_x, train_y, test_x, test_y, cv_metrics, param)

    logs.append(res)

    return logs

if __name__ == "__main__":


    filepath = "hw4_ex2_data.csv"

    data = []
    with open(filepath) as f:
        header = f.readline().strip().split(',')
        print('Header Categories:', end = " ")
        for i, category in enumerate(header): print(f"col {i}: {category}    ", end = "")
        for line in f:
            line = line.strip().split(',')
            line = [float(x) for x in line]
            data.append(line)
    data = np.array(data)
    x = data[:, :2]
    y = data[:, 2]
    print(f'\nThe shape of the data is: {data.shape}\nFeature shape: {x.shape}\nLabel shape: {y.shape}')
    print(f'Number of unique classes {len(np.unique(y))}')

    from sklearn.base import clone
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression()
    print(len(SimulateAL(x,y,'US', 2024, clone(clf), [], y, None, 5/x.shape[0], 0.5, TestLog, True, 3)))