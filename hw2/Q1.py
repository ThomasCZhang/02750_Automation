# Import packages
import multiprocessing as mp
import numpy as np

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_validate, train_test_split, StratifiedKFold
from sklearn.utils import resample

from Q3 import *

def QBC_Disagreement(pred_ys, type = 'KL'):
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

def UncertaintySample(
                    clf,
                    train_x: np.ndarray,
                    train_y: np.ndarray,
                    test_x: np.ndarray,
                    test_y: np.ndarray,
                    rng: np.random.Generator
                    ):
    """
    Returns the index of test data to add to train data using uncertainty sampling.
    Input:
    clf: the classifier
    train_x, train_y, test_x, test_y: The training and testing features and labels.

    Output:
    (np.ndarray) The utility vector that should be maximized to find the next index
    """
    active_model = clf.fit(train_x, train_y)
    prob_y = active_model.predict_proba(test_x)
    entropy = prob_y*np.log2(prob_y+1e-20)
    entropy = -np.sum(entropy, axis = 1)
    return entropy

def QuerryByCommittee(
                    clf, 
                    train_x: np.ndarray,
                    train_y: np.ndarray,
                    test_x: np.ndarray,
                    test_y: np.ndarray,
                    rng: np.random.Generator,
                    n_models: int=50,
                    ):
    """
    Returns the index of test data to add to train data using querry by committee.
    Input:
    clf: the classifier
    train_x, train_y, test_x, test_y: The training and testing features and labels.
    n_models: Number of models in the committee

    Output:
    (np.ndarray) The utility vector that should be maximized to find the next index
    """
    n_classes = len(np.unique(train_y))
    bootstraped_train_data = [resample(train_x, train_y, replace = True,  stratify = train_y) for _ in range(n_models)]
    bs_train_x = [tmp[0] for tmp in bootstraped_train_data]
    bs_train_y = [tmp[1] for tmp in bootstraped_train_data]

    models = [clone(clf) for _ in range(n_models)]
    for j in range(n_models):
        models[j].fit(bs_train_x[j], bs_train_y[j])
    prob_ys = np.hstack([models[j].predict_proba(test_x) for j in range(n_models)])
    prob_ys = prob_ys.reshape(test_x.shape[0], n_models, n_classes)
    disagreement = QBC_Disagreement(prob_ys)

    # next_idx = np.argmax(disagreement) # Index of next test data to add
    return disagreement

def MEROneSample(clf, 
            train_x: np.ndarray,
            train_y: np.ndarray,
            test_x: np.ndarray,
            test_y: np.ndarray,
            index) -> float:
    """
    Calculates the expected risk if we add one sample from the test set to the training set.
    Input:
    clf (sklearn predictor): The classifier from Sklearn
    train_x, train_y, test_x, test_y (np.ndarray): The training data, training labels, testing data, and testing labels.
    index (int): The index of the test sample that should be added to the training data to calculate expected risk.

    Output:
    (float) The expected risk from adding one sample to the training data. 
    """

    unique_labels = np.unique(train_y)
    n_classes = len(unique_labels)
    prob = clf.predict_proba(test_x[index:index+1])
    
    models = []
    newTrain_x = np.vstack((train_x, test_x[index, :]))
    for label in unique_labels:
        newTrain_y = np.append(train_y, label)
        models.append(clone(clf).fit(newTrain_x, newTrain_y))

    newTest_x = np.delete(test_x, index, axis = 0)

    prob_ys = np.hstack([models[i].predict_proba(newTest_x) for i in range(n_classes)])
    prob_ys = prob_ys.reshape(newTest_x.shape[0], n_classes, n_classes)
    
    score = np.sum(prob*np.sum(1 - np.max(prob_ys, axis = 2), axis = 0))
    return score
  

def ExpectedRiskMinimization(                    
                    clf, 
                    train_x: np.ndarray,
                    train_y: np.ndarray,
                    test_x: np.ndarray,
                    test_y: np.ndarray,
                    rng: np.random.Generator
                    ):
    """
    Returns the index of test data to add to train data using Expected Risk minimization.
    Input:
    clf: the classifier
    train_x, train_y, test_x, test_y: The training and testing features and labels.

    Output:
    (np.ndarray) The utility vector that should be maximized to find the next index
    """
    clf = clf.fit(train_x, train_y)
    if test_x.shape[0] > 1:
        er_scores = -np.array([MEROneSample(clf, train_x, train_y, test_x, test_y, i) for i in range(test_x.shape[0])])
        er_scores = er_scores + np.min(er_scores)
        return er_scores
    else:
        return np.array([1])

def UpdateDensity(test_x: np.ndarray, index: int, densities: np.ndarray):
    """
    If densities already exists then updates
    the densities to what they would be if data corresponding from the given index was not in the
    test data.
    Input:
    test_x (np.ndarray): the test data
    index (int): the index of test_x to remove

    Output:
    (np.ndarray) The density of each sample in test_x excluding the sample that was removed.
    """
    n_sample = test_x.shape[0]
    if n_sample == 1:
        densities = np.array([])
    else:
        dist_matrix = np.linalg.norm(test_x - test_x[index]) # Distance from the indexed sample to the all test_x samples.
        densities = (densities*n_sample - dist_matrix)/(n_sample-1)
        densities = np.delete(densities, index, axis = 0)
    return densities    

def CalculateDensity(test_x: np.ndarray,):
    """
    Calculates the density of the points in the test data. 
    Input:
    test_x (np.ndarray): The test data

    Output:
    (np.ndarray)  The density of each sample in test_x
    """
    n_sample = test_x.shape[0]
    res = np.array([np.sum(np.linalg.norm(test_x - test_x[i], axis = 1)) for i in range(n_sample)]) / n_sample
    return res

def DensitySampling(
                    clf, 
                    train_x: np.ndarray,
                    train_y: np.ndarray,
                    test_x: np.ndarray,
                    test_y: np.ndarray,
                    rng: np.random.Generator,
                    densities: np.ndarray,
                    beta: np.ndarray,
                    n_models: int=50,
):
    """
    Returns the index of test data to add to train data using density sampling with querry by committee.
    Input:
    clf: the classifier
    train_x, train_y, test_x, test_y: The training and testing features and labels.
    n_models: Number of models in the committee

    Output:
    (np.ndarray) The utility vector that should be maximized to find the next index
    """
    n_classes = len(np.unique(train_y))
    bootstraped_train_data = [resample(train_x, train_y, replace = True,  stratify = train_y) for _ in range(n_models)]
    bs_train_x = [tmp[0] for tmp in bootstraped_train_data]
    bs_train_y = [tmp[1] for tmp in bootstraped_train_data]

    models = [clone(clf) for _ in range(n_models)]
    for j in range(n_models):
        models[j].fit(bs_train_x[j], bs_train_y[j])
    prob_ys = np.hstack([models[j].predict_proba(test_x) for j in range(n_models)])
    prob_ys = prob_ys.reshape(test_x.shape[0], n_models, n_classes)
    disagreement = QBC_Disagreement(prob_ys)
    weighted_disagreement = densities*(disagreement)**beta

    # next_idx = np.argmax(weighted_disagreement)
    return weighted_disagreement

def NoStrategy(clf, 
            train_x: np.ndarray,
            train_y: np.ndarray,
            test_x: np.ndarray,
            test_y: np.ndarray,
            rng: np.random.Generator):
    """
    Returns a utility vector with 1 at some random index and 0 at all other indices.
    The size of the utility vector will be the number of smaples in test_x.
    """
    utility = np.zeros(test_x.shape[0])
    idx = rng.integers(0, test_x.shape[0])
    utility[idx] = 1
    return utility

def ChooseNextIndex(train_x, train_y, test_x, test_y, clf,
                    densities = None,
                    method = 'rand',
                    rng = None):
    """
    ChooseNextIndex: Chooses the next index from the test set to add to the training
    data for active learning

    Input:
    train-x (np.ndarray): The training data features
    train-y (np.ndarray): The training data labels
    test-x (np.ndarray): The testing data features
    test-y (np.ndarray): The testing data labels
    clf: The classifier
    densities: The densities of the test samples to be used in density based sampling.
    method (str): The active learning method to use.
        rand: No strategy
        US: Uncertainty sampling
        QBC: Querry by committee
        ERM: Expected Risk Minimization
        DBS: Density Based Sampling
        IWAL: Importance Weighted Active Learning
    rng (np.random.Generator): A seeded numpy random number generator.

    Output:
    The utility vector that should be maximized to find the next index
    """
    querry_strategies = {
        'rand': NoStrategy,
        'US': UncertaintySample,
        'QBC': QuerryByCommittee,
        'ERM': ExpectedRiskMinimization,
        'DBS': DensitySampling,
        'IWAL': NoStrategy
    }
    if method in querry_strategies:
        method_func = querry_strategies[method]
    else:
        raise Exception("Method not implemented")
    
    if method == 'DBS':
        beta = 0.7
        utility = method_func(clf, train_x, train_y, test_x, test_y, rng, densities, beta)
    else:
        utility = method_func(clf, train_x, train_y, test_x, test_y, rng)
    
    return utility

def UpdateLabeledData(train_x, train_y, test_x, test_y, clf,
                    modality = 'aggressive', threshold = 0.5, densities = None,
                    iwal_weights = None, method = 'rand',
                    b: int = 0, k: int = 50,  pmin: float = 0.1,
                    rng: np.random.Generator = None):
        """
        Updates the training data set with (up to) one instance from the test data set.
        Input:
        train-x (np.ndarray): The training data features
        train-y (np.ndarray): The training data labels
        test-x (np.ndarray): The testing data features
        test-y (np.ndarray): The testing data labels
        clf: Classifier
        modality (string): 
            'aggressive': Add the best sample available
            'mellow': Add the first sample that has utility score above some threshold
        threshold (float): The threshold for mellow active learning.
        densities: The densities of the test samples to be used in density based sampling.
        method (str): The active learning method to use.
            rand: No strategy
            US: Uncertainty sampling
            QBC: Querry by committee
            ERM: Expected Risk Minimization
            DBS: Density Based Sampling
            IWAL: Importance Weighted Active Learning
        b (int): The minimum size of the training data before beginning bootstrapping for IWAL.
        k (int): The number of bootstrapped datasets to create for IWAL.
        pmin (float): The minimum sampling probability for IWAL.
        rng (np.random.Generator): A seeded numpy random number generator.

        
        Output:
        train_x, train_y, test_x, test_y (np.ndarray): The updated training and testing sets. 
        """
        add_to_train = True
        utility = ChooseNextIndex(train_x, train_y, test_x, test_y,
                                   clf, densities = densities,
                                    method = method, rng = rng)

        if modality == 'aggressive':
            next_idx = np.argmax(utility)
        elif modality == 'mellow':
            utility = utility/np.sum(utility)
            next_idx = rng.choice(np.arange(utility.shape[0]), p=utility)
            
            # utility = utility/np.max(utility)
            # next_idx = np.argmax(utility >= threshold)


        if method == 'DBS':
            densities = UpdateDensity(test_x, next_idx, densities)
        if method == 'IWAL':
            pt = IWAL(clf, test_x[next_idx], train_x, train_y, iwal_weights, b, k, pmin, rng)
            if pt is None:
                add_to_train = False
            else:
                iwal_weights = np.append(iwal_weights, [pmin/pt])
            

        train_x, train_y, test_x, test_y = UpdateTrainTest(train_x, train_y, test_x, test_y, next_idx, add_to_train)

        return train_x, train_y, test_x, test_y, densities, iwal_weights

def UpdateTrainTest(train_x, train_y, test_x, test_y, index, add_to_train):
    """
    Takes the one sample from the test set and moves it to the training set.
    
    Input:
    train-x (np.ndarray): The training data features
    train-y (np.ndarray): The training data labels
    test-x (np.ndarray): The testing data features
    test-y (np.ndarray): The testing data labels
    index (int): The index of the sample to move.
    add_to_train (bool): Whether the test sample should be added to the training set or just deleted.

    Output:
    train_x, train_y, test_x, test_y (np.ndarray): The updated training and testing data sets
    """

    if add_to_train:
        train_x = np.vstack((train_x, test_x[index]))
        train_y = np.append(train_y, test_y[index])
    test_x = np.delete(test_x, index, axis = 0)
    test_y = np.delete(test_y, index, axis = 0)

    return train_x, train_y, test_x, test_y

def SimulateAL_MP(x: np.ndarray, y: np.ndarray,
                method: str, seed: int,
                conn: mp.connection.PipeConnection = None,
                modality: str = 'aggressive',
                threshold: float = 0.5,
                k: int = 50,
                pmin: float = 0.1
                ):
    """
    Simulates active learning using the designated active learning method.
    Input:
    x (np.ndarray): data (features)
    y (np.ndarray): label of the data
    method (str): The active learning method to use.
        rand: No strategy
        US: Uncertainty sampling
        QBC: Querry by committee
        ERM: Expected Risk Minimization
        IWAL: Importance Weighted Active Learning
    seed (int): The random seed for splitting the data into train/test.
    modality (str): aggressive or mellow
    conn (mp.connection.PipeConnection): A connector that sends data.
    threshold (float): value between 0 and 1. Used for thresholding when using mellow active learning.
    k (int): Number of bootstrapped samples to use.
    pmin (float): Minimum sampling chance to be used with IWAL.
    

    Output:
    Returns the accuracy and loss logs across the iterations of the active learning simulation
    """

    # clf = SVC(probability=True)
    preprocess = StandardScaler()
    x = preprocess.fit_transform(x)
    clf = LogisticRegression()
    rng = np.random.default_rng(seed) # Set the seed for the random number generator
    np.random.seed(seed)
    b = x.shape[0]//5

    if method != 'IWAL':
        train_x, test_x, train_y, test_y = train_test_split(x,y, train_size = 0.2,
                                                             shuffle=True, stratify = y, random_state = seed)
        iwal_weights=  None
    else:
        # Since first b samples are added for sure
        # We can just use train_test_split to preallocate the starting b labeled points.
        train_x, test_x, train_y, test_y = train_test_split(x,y, train_size = b,
                                                             shuffle=True, stratify = y, random_state = seed)
        iwal_weights = np.ones(train_x.shape[0])*pmin

    densities = None
    if method == 'DBS':
        densities = CalculateDensity(test_x)
    # iwal_weights = None
    # if method == 'IWAL':
    #     iwal_weights = np.array([])
        
    acc_log, loss_log, size_log = [], [], []
    n_iters = test_y.shape[0] # Number of samples in the test set
    # print(n_iters)
    cv = StratifiedKFold(n_splits = 5, shuffle=True, random_state=seed)

    for _ in range(n_iters):
        # print(f'{_}/{n_iters}', end = '\r')
        if train_x.shape[0] >= int(x.shape[0]*0.2):
            if method == 'IWAL':
                res = cross_validate(clf, train_x, train_y,
                    params = {'sample_weight': iwal_weights},
                    cv = cv, scoring = ['accuracy', 'neg_log_loss'])            
            else:
                res = cross_validate(clf, train_x, train_y,
                                    cv = cv, scoring = ['accuracy', 'neg_log_loss'])
            
            acc_log.append(np.mean(res['test_accuracy']))
            loss_log.append(-np.mean(res['test_neg_log_loss']))
            size_log.append(train_x.shape[0])

        train_x, train_y , test_x, test_y, densities, iwal_weights = UpdateLabeledData(train_x, train_y, test_x, test_y, clf,
                    modality = modality, threshold = threshold,
                    densities = densities, iwal_weights=iwal_weights,
                    method = method, b=b, k = k , pmin = pmin, rng = rng)
    
    if method == 'IWAL':
        res = cross_validate(clf, train_x, train_y,
                    params = {'sample_weight': iwal_weights},
                    cv = cv, scoring = ['accuracy', 'neg_log_loss'])            
    else:
        res = cross_validate(clf, train_x, train_y,
                            cv = cv, scoring = ['accuracy', 'neg_log_loss'])

    # print('\n I MADE IT!')
    if conn is not None:
        conn.send((acc_log, loss_log, size_log))
    else:
        return (acc_log, loss_log, size_log)
