import numpy as np
from argparse import ArgumentParser
import itertools

# Objectives. I've added a negative infront of the objectives that are supposed to be maximized. So
# instead of maximizing them in the parent function, we can minimize. This is so that I can 
# swap out any of the objectives easily.
def A_optimal(X):
    """
    Trace of inverse of covariance matrix.
    """
    try:
        result = np.trace(np.linalg.inv(X.T@X))
    except:
        print("No inverse. Using Moore-Penrose pseudo inverse.")
        result = np.trace(np.linalg.pinv(X.T@X))
    return result

def D_optimal(X):
    """
    Determinant of covariance matrix
    """
    return -np.linalg.det(X.T@X)

def G_optimal(X):
    """
    Expected variance of output
    """
    try:
        result = np.max(np.diag(X@np.linalg.inv(X.T@X)@X.T))
    except:
        print("No inverse. Using Moore-Penrose pseudo inverse.")
        result = np.max(np.diag(X@np.linalg.pinv(X.T@X)@X.T))
    return result

def E_optimal(X):
    """
    Eiganvalue of covariance matrix
    """
    return -np.min(np.linalg.eigvals(X.T@X))

def DOE(X, Y, k, seed, cap, threshold, metric):
    """
    Design of experiments
    Input:
    X (np.ndarray): A N x d matrix. N = number of samples. d = dimension of samples.
    Y (np.ndarray): A N x 1 matrix. N = number of samples.
    k (int): The number of samples to take from X.
    seed (int): The seed for the numpy random generator
    cap (int): The maximum number of cycles to run DOE.
    metric (function): The function for calculating the metric to be maximized.

    Output:
    samps (np.ndarray): A k x d matrix. The chosen samples from X.
    """
    rng = np.random.default_rng(seed)
    start_k = rng.choice(np.arange(X.shape[0]),size=k,replace=False,axis=0,shuffle=True)
    mask = np.array([False for _ in range(X.shape[0])], dtype = bool)
    mask[start_k] = True
    # print(mask)
    # Select random set of starting samples
    current_X = X[start_k]
    current_Y = Y[start_k]
    i = 0

    current_k = start_k
    current_score = metric(current_X)
    # print(current_score)
    delta = np.inf
    while i < cap and np.abs(delta) > threshold:
        # Choose the sample in the training set to replace
        replace_idx = rng.integers(0, k)
        mask = np.array([True for _ in range(X.shape[0])], dtype = bool)
        mask[current_k] = False
        viable_idx = np.arange(0, X.shape[0])[mask]

        # Weight the samples not in the training set.
        weights = np.zeros(viable_idx.shape[0])
        for idx, j in enumerate(viable_idx):
            new_k = current_k.copy()
            new_k[replace_idx] = j
            new_x = X[new_k]
            weights[idx] = metric(new_x)

        # Calculates the weights
        normalize_value = np.sum(weights)
        weights /= normalize_value
        if normalize_value > 0:
            weights = 1/weights
            weights /= np.sum(weights)

        # new_idx = rng.choice(viable_idx, p=weights)
        new_idx = viable_idx[np.argmax(weights)]
        new_k = current_k.copy()
        new_k[replace_idx] = new_idx
        new_x = X[new_k]
        new_score = metric(new_x)
        delta = new_score-current_score
        if delta < 0:
            current_k = new_k
            current_score = new_score
        else:
            if rng.random() < weights[np.where(viable_idx==new_idx)]:
                current_k = new_k
                current_score = new_score

        i += 1
    # print(f'DOE Best {current_score}')
    
    return current_score, current_k


def CheckAll(X, k, model):    
    best_score = np.inf
    for start_k in itertools.combinations(np.arange(X.shape[0]), k):
        start_k = np.array(start_k)
        curr_X = X[start_k]
        score = model(curr_X)
        if score < best_score:
            best_score = score
            best_idxs = start_k.copy()
    return best_score, best_idxs
        

def ReadExercise1(fp: str):
    """
    Reads in data for exercise 1.
    Input:
    fp (str): Filepath to exercise 1 data location.
    """
    data = []
    with open(fp) as f:
        f.readline()
        i = 1
        for line in f:
            i += 1
            temp = line.strip().split(',')
            temp = [float(x) for x in temp]
            data.append(temp)

    data = np.array(data)
    feat = data[:,:-1]
    labels = data[:,-1]
    return feat, labels
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('filepath', type=str, help="Path to the dataset")
    args = parser.parse_args()
    filepath = args.filepath
    X, Y = ReadExercise1(filepath)

# ALICE But I don't know how to get pTest
# def MakeU(x, ptest, order):
#     """
#     Makes U in the ALICE algorithm.
#     Assumes linear basis function. Order is the order of the basis function.
#     Input:
#     x (np.ndarray): The sample
#     ptest (float): The probability assigned to x by the model. P_test(x)
#     order (int): The order of the linear basis function.
#     """
#     U = np.empty((order+1, order+1))
#     for i in range(order+1):
#         for j in range(i+1, order+1):
#             U[i,j] = CalculateUij(x, i, j, ptest)
#     return U

# def CalculateUij(x,i,j,ptest):
#     """
#     Calculates the value of U[i,j] for unlabeled sample x according to ALICE algorithm.
    
#     Input:
#     x (np.ndarray): The sample being analyzed. A 1 x d vector where d is the dimensionality of the data.
#     i (int): The powr of the first basis function
#     j (int): The power of the second basis function
#     ptest (float): The probability assigned to x by the model. P_test(x).
#     """
#     return np.dot(x**i, x**j)*ptest
    
# def ALICE(dataset, model, order, k):
#     """
#     Input:
#     dataset: The dataset we need to choose samples from. N x k matrix. N samples, k features.
#     model: The predictor model.
#     k: The number of samples to take from the dataset.
#     """
#     N, k = dataset.shape

#     U = np.zeros((order+1, order+1))
#     for i in range(N):
#         sample = N[i, :]
#         ptest = model.predict_proba(sample)
#         U += MakeU(sample, ptest, order)
    
    
#     pass