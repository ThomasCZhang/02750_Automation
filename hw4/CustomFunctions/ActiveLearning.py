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
from DiversitySampling import *
from VariableCostUncertaintySampling import *
from LowestCost import *
from RandomCost import *

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

def ChooseActiveLearningMethod(x: np.ndarray, y: np.ndarray,
                                cost: np.ndarray=None,
                                method: str='US',
                                seed: int=1,
                                stratify_data: np.ndarray=None,
                                init_method: str=None,
                                init_frac: float=None,
                                aggressive: bool = True,) -> any:
    """
    Creates the active learning updater class object based on what the method is.
    Input:
    x (np.ndarray): data (features).
    y (np.ndarray): label of the data.
    cost (np.ndarray): The cost of getting the label from the oracle.
    method (str): The active learning method to use.
        PL: Passive Learning
        US: Uncertainty sampling
        QBC: Querry by committee
        ERM: Expected Risk Minimization
        DBS: Density Sampling
        IWAL: Importance Weighted Active Learning
        DVS: Diversity Sampling
        VCUS: Variable Cost Uncertainty Sampling
        LC: Lowest Cost
        RC: Random Cost
    seed (int): A seed for the numpy random number generator.
    stratify_data (np.ndarray): the data to stratify on if using train-test split
    init_method (str): if "DOE" uses DOE to initialize. Otherwise does a stratified train-test split.
    init_frac (float): Value between 0 and 1. Proportion of data (labeled data) to use for initial model.
    aggressive (bool): If True uses aggressive active learning. Otherwise mellow aggressive learning. Default is True.
    """
    np.random.seed(seed)

    x = np.hstack((cost.reshape(-1, 1), x)) if cost is not None else x
    train_x, test_x, train_y, test_y = InitialSplit(x, y, seed, stratify_data=stratify_data,
                                    init_method = init_method, init_frac = init_frac)
    if cost is not None:
        train_x = train_x[:, 1:]
        cost, test_x = test_x[:,0], test_x[:, 1:]
    
    strategy_dict = {
        "PL": PassiveLearning,
        "US": UncertaintySampling,
        "QBC": QueryByCommittee,
        "ERM": ExpectedRiskMinimization,
        "DBS": DensitySampling,
        "IWAL": IWAL,
        "DVS": DiversitySampling,
        "VCUS": VariableCostUncertaintySampling,
        "LC": LowestCost,
        "RC": RandomCost,
    }
    active_learning_method = strategy_dict[method](train_x, train_y, test_x, test_y, cost, aggressive)
    return active_learning_method

def ActiveLearningOneLoop(active_learning_method: any,
                        method: str,
                        clf: any,
                        cv_metrics: list[str],
                        logging_func: any,
                        logs: list,
                        size_log: list,
                        batch_size: int=1) -> None:
    """
    Performs one iteration of the active learning loop
    Input:
    active_learning_method (ActiveLearningUpdater): Custom active learning updater class.
    method (str): The method used for active learning.
        PL: Passive Learning
        US: Uncertainty sampling
        QBC: Querry by committee
        ERM: Expected Risk Minimization
        DBS: Density Sampling
        IWAL: Importance Weighted Active Learning
        DVS: Diversity Sampling
    clf (sklearn model): The sklearn model.
    cv_metrics (list[str]): The list of metrics to measure using sklearn cross validation
    logging_func (function): The custom function used to get the logging metrics.
    log (list): For storing whatever metric (accuracy, loss, etc.) that needs to be reported
    size_log (list): For storing the size of the labeled dataset.
    batch_size (int): The number of samples to add per iteration.
    """
    train_x, train_y, test_x, test_y = active_learning_method.GetData()
    param = {'sample_weight': active_learning_method.weights} if method == 'IWAL' else None

    res = logging_func(clf, train_x, train_y, test_x, test_y, cv_metrics, param)
    if active_learning_method.cost is not None: res.append(active_learning_method.spent)
    logs.append(res)
    size_log.append(train_x.shape[0])
    active_learning_method.Update(clf, batch_size=batch_size)

def SimulateAL(x: np.ndarray, y: np.ndarray,
                cost: np.ndarray=None,
                method: str='US',
                seed: int=1,
                clf: any=None ,
                cv_metrics: list[str]=None,
                stratify_data: np.ndarray=None,
                init_method: str=None,
                init_frac: float=None,
                end_frac: float=None,
                budget: float=None,
                logging_func: any=None,
                aggressive: bool = True,
                batch_size: int=1,
                ):
    """
    Simulates active learning using the designated active learning method.
    Input:
    x (np.ndarray): data (features)
    y (np.ndarray): label of the data
    cost (np.ndarray): The cost of getting the label from the oracle.
    method (str): The active learning method to use.
        PL: Passive Learning
        US: Uncertainty sampling
        QBC: Querry by committee
        ERM: Expected Risk Minimization
        DBS: Density Sampling
        IWAL: Importance Weighted Active Learning
        DVS: Diversity Sampling
        VCUS: Variable Cost Uncertainty Sampling
        LC: Lowest Cost
        RC: Random Cost
    seed (int): The random seed for the numpy random generator   
    clf (SKlearn classifier): The classifier to use
    cv_metrics (list[str]): Input arguments to the scoring argument of sklearns cross validation.
    stratify_data (np.ndarray): The data to use for stratifying when performing train_test_split.
    init_method (str): if "DOE" uses DOE to initialize. Otherwise does a stratified train-test split.
    init_frac (float): Value between 0 and 1. Proportion of data to use for initial model.
    end_frac (float): value between 0 and 1. Proportion of data to use for final model after active learning.
    budget (float): The budget allocated to active learning.
    logging_func (function): used to calculate the stuff to log. 
    aggressive (bool): Use aggressive active learning. Default is True.
    batch_size (int): The number of samples to add at once. Default is 1.

    Output:
    Returns the accuracy and loss logs across the iterations of the active learning simulation
    """

    preprocess = StandardScaler()
    x = preprocess.fit_transform(x)    
    active_learning_method = ChooseActiveLearningMethod(x, y, cost, method, seed,
                                                        stratify_data, init_method, init_frac, aggressive)
    size_log, logs = [], []
    n_iters = int(np.ceil((int(end_frac*x.shape[0])-active_learning_method.x_lab.shape[0])/batch_size))
    for _ in range(n_iters):
        ActiveLearningOneLoop(active_learning_method, method, clf, cv_metrics, logging_func, logs, size_log, batch_size)
        if cost is not None and active_learning_method.spent > budget:
            break

    train_x, train_y, test_x, test_y = active_learning_method.GetData()
    param = {'sample_weight': active_learning_method.weights} if method == 'IWAL' else None

    np.random.seed(seed)
    res = logging_func(clf, train_x, train_y, test_x, test_y, cv_metrics, param)
    if active_learning_method.cost is not None: res.append(active_learning_method.spent)
    logs.append(res)
    return logs
