import numpy as np
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler

def Similarity(X_lab: np.ndarray, X_unlab: np.ndarray, sigma: float=None) -> np.ndarray:
    """
    X_lab (np.ndarray): N by d matrix. The unlabeled data. N = number of labeled samples. d is the dimension of one sample.
    X_unlab (np.ndarray): M by d matrix. The labeled data. M = number of unlabled samples. d is the dimension of one sample.
    sigma (float): Scaling factor

    Output:
    similarity_matrix (np.ndarray): A N by N similarity matrix.
    """
    x = np.concatenate((X_lab, X_unlab), axis = 0)
    num_samples = x.shape[0]

    similarity_matrix = np.zeros((num_samples,num_samples))
    for i in range(num_samples-1):
        delta_X = np.sum(np.power(x[i+1:,:] - x[i,:],2), axis = 1)
        similarity_matrix[i,i+1:] = delta_X[np.newaxis, :]
        similarity_matrix[i+1:,i] = delta_X[np.newaxis, :]

    if sigma is None:
        # sigma = np.sqrt(np.max(similarity_matrix))
        sigma = np.std(x)

    similarity_matrix /= -sigma**2

    return np.exp(similarity_matrix)

def BuildGraphAsMatrix(similarity: np.ndarray, t:float=None, n:int=10) -> np.ndarray:
    """
    Input:
    similarity (np.ndarray): N x N similarity matrix. N = number of unlabeled samples.
    t (float): Cutoff for whether edge in graph exists. If W[i,j] > t then the edge exists. Otherwise there is no 
        edge connecting node i to node j.
        If t is None. Then instead of using a cutoff. The largest n values in a row are kept while the rest are set to
        0.
    n (int): The minimum number of neighbors each vertex will have. Only used if t is None.

    Output:
    g (np.ndarray): A weight matrix representing the graph. G[i,j] = weight of edge (i,j). 0 if there is no 
        edge connecting i and j.
    """
    num_samples = similarity.shape[0]
    g = similarity.copy()
    if t is None:
        mask = np.array(np.ones(g.shape), dtype = bool)
        col_idxs = np.argsort(g,axis = 1)[:, :n].flatten()
        row_idxs = np.arange(num_samples).repeat(n)
        mask[row_idxs, col_idxs] = False
        mask[col_idxs, row_idxs] = False # Make the weight matrix symmetrical.
        g[mask] = 0
    else:
        g[g < t] = 0
    return g

def ConstructD(weights: np.ndarray) -> np.ndarray:
    """
    Constructs the matrix D for ZLG. The value on d[i] = sum over j of W[i,j]
    Input:
    weights (np.ndarray): Weight matrix used to create the D matrix.

    Output:
    diagonal (np.ndarray): n x n diagonal matrix. 
    """
    return np.diag(np.sum(weights, axis = 1))

def ConstructLaplacian(weights: np.ndarray, d_matrix: np.ndarray) -> np.ndarray:
    """
    Constructs the Laplacian matrix for ZLG.
    Input: 
    weights (np.ndarray): N x N matrix. Weight matrix of the graph.
    d_matrix (np.ndarray): N x N diagonal matrix. Value of diagonal is sum of row in W.

    Output:
    (np.ndarray): The laplacian matrix D - W
    """
    return d_matrix - weights 

def SolveHarmonics(laplacian: np.ndarray, y_lab: np.ndarray) -> float:
    """
    Estimates the harmonic of unlabled data. These are also the predicted labels of the unlabeled data.
    Inputs:
    laplacian (np.ndarray): N x N matrix. The laplacian matrix. N = total number of samples (labeled and unlabeled).
    y_lab (np.ndarray): M length array. The labels of the labeled data. M = number of labeled samples.

    Outputs:
    harmonics (np.ndarray). A (N-M) length array. The estimated mean label of the unlabeled data.
    inv_laplacian_unlab (np.ndarray). A (N-M) x (N-M) matrix. The laplacian submatrix corresponding to unlabeled data..
    """

    num_labeled = y_lab.shape[0]
    laplace_uu = laplacian[num_labeled:, num_labeled:]

    try:
        harmonics = -np.linalg.inv(laplace_uu)@laplacian[num_labeled:, :num_labeled]@y_lab
    except:
        harmonics = -np.linalg.pinv(laplace_uu)@laplacian[num_labeled:, :num_labeled]@y_lab

    return harmonics, laplace_uu

def CalculateRisk(prob_matrix: np.ndarray) -> float:
    """
    Calculates the risk given a vector of probabilities.
    Input:
    prob_matrix (np.ndarray): N x N matrix. A vector of probabilities that the label is 1. Each column corresponds to the
        predictions from one model.
    
    Output:
    (float): N length array. The risks based on each probability vector. Each risk corresponds to one of the models.
    """
    probs = np.stack((prob_matrix, 1-prob_matrix),axis = -1)
    risk = np.sum(np.min(probs, axis = -1), axis = 0)
    return risk

def UpdateProbs(probs: np.ndarray, inv_laplace_uu: np.ndarray, lab: int) -> np.ndarray:
    """
    Updates the probabilties for the unlabeld samples under the assumtion that unlabeled sample number "k" is added to the
    labeled data with the label "lab".
    Input:
    probs (np.ndarray): N length array. The original probabilities for the unlabeled data.
    inv_laplace_uu (np.ndarray): N x N array. The pseudo-inverted unlabeled portion of the Laplacian matrix.
    lab (int): 0 or 1. The label to be assigned to the unlabeled data.
    
    Output:
    (np.ndarray): N x N probability array. Each column is the probabilities for one model. The new probabilities.
    """

    new_probs = probs[:, np.newaxis] + (lab - probs)*inv_laplace_uu/inv_laplace_uu.diagonal() 

    # if np.isnan(new_probs).any():
    #     for val in inv_laplace_uu.diagonal():
    #         print(val)
    #     raise Exception
    return new_probs

def CalculateUpdatedRisks(probs: np.ndarray, inv_laplace_uu: np.ndarray) -> np.ndarray:
    """
    Calculates the risk of adding an unlabled sample to the labeled data for all unlabeled samples.
    Input:
    probs (np.ndarray): N length array. The predictions of the current unlabled data.
    inv_laplace_uu (np.ndarray): The pseudo-inverted unlabeled portion of the laplacian matrix.

    Ouput:
    (np.ndarray): Length N array. The estimated risk for each of the unlabled samples to the labled set.
    """
    new_probs_0 = UpdateProbs(probs, inv_laplace_uu, 0)
    new_probs_1 = UpdateProbs(probs, inv_laplace_uu, 1)
    risk_0 = CalculateRisk(new_probs_0)
    risk_1 = CalculateRisk(new_probs_1)
    risks = (1-probs)*risk_0 + probs*risk_1
    return risks

def InitFromTestAndTrain(train_x, train_y, test_x, t=None, n=10):
    """
    Initializes the harmonics and the unlabeled portion of the laplacian matrix.
    Input:
    train_x (np.ndarray): Training data. 
    train_y (np.ndarray): Training labels.
    test_x (np.ndarray): Test data.
    t (float): Cutoff for whether edge in graph exists. If W[i,j] > t then the edge exists. Otherwise there is no 
        edge connecting node i to node j.
        If t is None. Then instead of using a cutoff. The largest n values in a row are kept while the rest are set to
        0.
    n (int): The minimum number of neighbors each vertex will have. Only used if t is None.

    Output:
    probs (np.ndarray): The harmonics (probabilities) of the unlabeled data.
    laplace_uu: The inverse of the unlabeled portion of the laplacian.
    """
    similarity = Similarity(train_x, test_x)
    weights = BuildGraphAsMatrix(similarity, t, n)
    d_matrix = ConstructD(weights)
    laplacian = ConstructLaplacian(weights,d_matrix)
    probs, laplace_uu = SolveHarmonics(laplacian,train_y)
    return probs, laplace_uu

def ZLGOneIter(train_x: np.ndarray,
                train_y: np.ndarray,
                test_x: np.ndarray,
                test_y: np.ndarray,
                probs: np.ndarray,
                laplace_uu:np.ndarray) -> tuple[np.ndarray]:
    """
    Calculates the risk of adding an unlabled sample to the labeled data for all unlabeled samples.
    Input:
    train_x (np.ndarray): N x d matrix. Training features.
    train_y (np.ndarray): N length array. Training labels.
    test_x (np.ndarray): M x d matrix. Test features.
    test_y (np.ndarray): m length array. Test labels.
    probs (np.ndarray): M length array. The predictions of the current unlabled data.
    laplace_unlab (np.ndarray): The unlabeled portion of the laplacian matrix.

    Ouput:
    tuple[np.ndarray]: Updated train_x, train_y, test_x, test_y, probs, and laplace_uu.
    """
    try:
        inv_laplace_uu = np.linalg.inv(laplace_uu)
    except:
        inv_laplace_uu = np.linalg.pinv(laplace_uu)

    inv_laplace_uu = inv_laplace_uu + np.diag(1e-30*np.ones(inv_laplace_uu.shape[0]))

    risks = CalculateUpdatedRisks(probs, inv_laplace_uu)
    min_idx = np.argmin(risks) # Minimizing expected risk

    train_x = np.vstack((train_x, test_x[min_idx]))
    train_y = np.append(train_y, test_y[min_idx])
    test_x = np.delete(test_x, min_idx, axis = 0)
    test_y = np.delete(test_y, min_idx, axis = 0)

    # return train_x, train_y, test_x, test_y, probs, laplace_uu
    return train_x, train_y, test_x, test_y

def CrossValidate(clf, train_x, train_y, test_x, test_y, cv_metrics):
    """
    Log the results from cross-validating the training set data.
    """
    cv_res = cross_validate(clf, train_x, train_y, scoring = cv_metrics)
    res = [np.mean(cv_res['test_'+cv_metric]) for cv_metric in cv_metrics]

    return res

def SimulateZLG(x: np.ndarray,
                y: np.ndarray,
                seed:int,
                clf: any,
                cv_metrics: list[str],
                init_frac:float,
                end_frac:float,
                t: float = None,
                n: int = 10) -> tuple:
    """
    Calculates the risk of adding an unlabled sample to the labeled data for all unlabeled samples.
    Input:
    x (np.ndarray) N x d matrix. The feature data.
    y (np.ndarray) N length array. The labels.
    seed (int): rng seed.
    clf (sklearn classifier): The classifier to use.
    cv_metrics (list[str]): The metrics to record for cross_validation.
    init_frac (float): The starting percentage of the training data.
    end_frac (float): The ending percentage of the training data.
    t (float): Cutoff for whether edge in graph exists. If W[i,j] > t then the edge exists. Otherwise there is no 
        edge connecting node i to node j.
        If t is None. Then instead of using a cutoff. The largest n values in a row are kept while the rest are set to
        0.
    n (int): The minimum number of neighbors each vertex will have. Only used if t is None.

    Ouput:
    Tuple of lists. Each list is a record of the scores at each iteration.
    """
    preprocess = StandardScaler()
    x = preprocess.fit_transform(x)

    train_x, test_x, train_y, test_y = train_test_split(x,y,
                                        train_size=init_frac,
                                        shuffle=True,
                                        stratify=y,
                                        random_state = seed) 

    size_log, logs = [], []
    n_iters = int(end_frac*x.shape[0])-train_x.shape[0]
    for i in range(n_iters):
        probs, laplace_uu = InitFromTestAndTrain(train_x, train_y, test_x, t, n)
        res = CrossValidate(clf, train_x, train_y, test_x, test_y, cv_metrics)
        logs.append(res)
        size_log.append(train_x.shape[0])

        train_x, train_y, test_x, test_y = ZLGOneIter(
            train_x, train_y, test_x, test_y, probs, laplace_uu
        )

    np.random.seed(seed)
    res = CrossValidate(clf, train_x, train_y, test_x, test_y, cv_metrics)
    logs.append(res)
    
    return logs

def ReadExercise2Data(filepath):
    """
    Input:
    filepath (str): The path to the exercise 2 data. 

    Output:
    feat (np.ndarray): The data features as a numpy array.
    labels (np.ndarray): The data labels as a numpy array.
    """
    data = []
    with open(filepath) as f:
        f.readline() # Skip over header line.
        for line in f:
            line = line.strip().split(',')
            line = [float(x) for x in line]
            data.append(line)
    data = np.array(data)
    feat, labels = data[:,:-1], data[:,-1]
    return feat, labels

if __name__ == "__main__":
    feat, labels = ReadExercise2Data('ex2_data.csv')

    # filepath = 'ex2_data.csv'
    # x, y = ReadExercise2Data(filepath)

    # from sklearn.linear_model import LogisticRegression
    # clf = LogisticRegression()
    # cv_metric = ['neg_log_loss','accuracy']
    # SimulateZLG(x,y,2024,clf,cv_metric,0.3,1.0,0.5)
    
    # Testing Similarity Function
    # test_var = Similarity(feat)
    # print(test_var)
    # for i in range(test_var.shape[0]):
    #     if test_var[i,i] != 1:
    #         print('Diagonal not all ones.')

    # # Testing Graph Building Functions:
    # print(test_var.max())
    # test_dict = BuildGraphAsAdjacencyDict(test_var, 0.9)
    # for key in test_dict:
    #     print(test_dict[key])
    
    # test_matrix = BuildGraphAsMatrix(test_var, 0.9)
    # print(test_matrix)

#######
## Unused functions

# def GetMultivariateMeanAndCovariance(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Input:
#     x (np.ndarray): The data used to define the multivariate gaussian distribution
#         N x d matrix. N = number of samples. d = dimensions of data.

#     Output:
#     mean (np.ndarray): d dimensional array. The mean of each dimension.
#     cov (np.ndarray): d x d matrix. The covariance matrix.
#     """
#     cov = x.T @ x
#     mean = np.mean(x, axis = 0)
#     return mean, cov    

# def MultivariateGaussian(y: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
#     """
#     Input:
#     y (np.ndarray): A d dimensional array. The sample to calculate the multivariate gaussian probability.
#     mean (np.ndarray): A d dimensional array. The mean of the multivariate gaussian. 
#     cov (np.ndarray): A d x d matrix. The covariance matrix of the multivariate gaussian. 

#     Output:
#     prob: The probability of selecting y based on the multivariate gaussian distribution defined by the given 
#     mean and cov. 
#     """
#     n = y.shape[0]
#     det = np.linalg.det(cov)
#     try:
#         precision = np.linalg.inv(cov)
#     except:
#         print("Covariance matrix has no inverse (for some reason)???")
#         precision = np.linalg.pinv(cov)
#     delta = y - mean
#     delta = delta[:, np.newaxis]
    
#     prob = 1/(np.power(2*np.pi, n)*det)*np.exp(-0.5*delta.T @ precision @ delta)
#     return prob

# def ConstructEnergy(y: np.ndarray, L: np.ndarray) -> float:
#     """
#     Input:
#     Y (np.ndarray): d x 1 matrix. d = dimension of the samples 
#     L (np.ndarray): N x N laplacian matrix
    
#     Output:
#     energy (float): The energy used by the ZLG algorithm.
#     """
#     if y.ndim < 2:
#         y = y[:, np.newaxis]
#     return y.T @ L @ y

# def GenerateEnergies(Y: np.ndarray, L: np.ndarray) -> np.ndarray:
#     """
#     Input:
#     Y (np.ndarray): N x d matrix. d = dimension of the samples.
#     L (np.ndarray): N x N laplacian matrix
    
#     Output:
#     energy (float): The energy used by the ZLG algorithm.
#     """
#     Ey = [ConstructEnergy(x,L) for x in Y]
#     Ey = np.array(Ey)
#     return Ey

# def CalculateProbabilities(Ey: np.ndarray, Beta: float) -> np.ndarray:
#     """
#     Input:
#     Ey (np.ndarray): Energy. N x 1 matrix. 
#     Beta (float): Inverse temperature parameter
    
#     Output:
#     probabilities (np.ndarray): The probability of each sample in Y.
#     """
#     probabilities = np.exp(-Beta*Ey)
#     Zbeta = np.sum(probabilities)
#     probabilities = 1/Zbeta*probabilities
#     return probabilities
    
# def BuildGraphAsAdjacencyDict(W: np.ndarray, t:float) -> dict[int, list[tuple[int, float]]]:
#     """
#     Input:
#     W (np.ndarray): N x N similarity matrix. N = number of unlabeled samples.
#     t (float): Cutoff for whether edge in graph exists. If W[i,j] > t then the edge exists. Otherwise there is no 
#         edge connecting node i to node j.

#     Output:
#     G (dict[int, list[tuple[int, float]]]): An adjacency dictionary representing the graph. 
#         Keys correspond to a node in the graph.
#         Values are list of 2-element tuples. Each tuple represents an edge from the key to the first value in the tuple.
#         The weight of the edges are the second value in the tuple.
#     """
#     N = W.shape[0]
#     G = {key: [] for key in range(N)}

#     for i in range(N-1):
#         for j in range(i+1, N):
#             if W[i,j] > t:
#                 G[i].append((j, W[i,j]))
#                 G[j].append((i, W[i,j]))

#     return G