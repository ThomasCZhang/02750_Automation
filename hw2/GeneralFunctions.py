import numpy as np

def ReadHeartFailureData(filepath):
    all_data = np.loadtxt(filepath, delimiter=',',skiprows = 1)
    x = all_data[:, :-1]
    y = all_data[:, -1]
    return x,y


def ReadFibrosisData(filepath):
    all_data = np.loadtxt(filepath, delimiter=',')
    x = all_data[:, 1:]
    y = all_data[:, 0]
    return x,y

def ReadDiabetesData(filepath):
    """
    Read in the dataset for "Early stage diabetes risk prediction"
    """
    conversion_dict = {
        'Male': 0.,
        'Female': 1.,
        'No': 0.,
        'Yes': 1.,
        'Negative': 0.,
        'Positive': 1.
    }
    x, y = [], []
    with open(filepath) as f:
        column_labels = f.readline().strip().split(',')
        for row in f:
            row = row.strip().split(',')
            x.append([float(row[0])] + [conversion_dict[x] for x in row[1:]])
            y.append([conversion_dict[row[-1]]])
    x, y = np.array(x), np.array(y)
    return x, y

# def Bootstrap(*args, m, rng = None):
#     """
#     Bootstrap takes multiple iterables and samples them with replacement for bootstrapping.
#     *args (np.ndarray): The arrays to bootstrap. 
#     m (int): The number of new training sets to create.
#     rn (np.random.Generator): The random number generator to use.

#     Return:
#     A list of arrays. Each array will correspond to one of the arrays in args with an extra first dimension.
#     That extra first dimension will be m. e.g. If an array in args is r x c, then the corresponding array in
#     the output will be m x r x c.
#     """
#     for iterable in args:
#         assert type(iterable) == np.ndarray, '*args must be of type np.ndarray!'
#     data_length = args[0].shape[0]
#     for iterable in args:
#         assert(len(iterable) == data_length), 'Expected all arrays in *args to have the same number of samples (rows)!'

    
#     if rng:
#         indices = rng.integers(0, data_length, m*data_length)
#     else:
#         indices = np.random.randint(0, data_length, m*data_length)
    
#     new_datasets = [arr[indices].reshape(m, *arr.shape) for arr in args]
#     return new_datasets

# def SplitData(*args, split_percent, rng = None):
#     """
#     SplitData collectively splits the data (no replacement).
#     E.g. If given 2 arrays it will split the first and second array 
#     the same way.
#     *args (np.ndarray): arrays the should be split. The arrays should all have the same size first dimension.
#     split_percent (float or iterable of floats): How large each split should be.
#     """
#     # args = [arr.copy() for arr in args] 
#     num_samples = args[0].shape[0] # The number of samples

#     p_sum = np.sum(split_percent) # The sum of the probabilities.
#     if np.isclose(p_sum, 1, atol=1e-2):
#         p_sum = 1.

#     for arr in args:
#         assert type(arr) == np.ndarray, '*args must be of type np.ndarray!'

#     assert p_sum <= 1., 'Percentage of splits must sum to 1 or less.'
    
#     shuffled_idx = np.arange(num_samples) # Create a range of numbers to use for random sampling.
#     if rng:
#         rng.shuffle(shuffled_idx)
#     else:
#         np.random.shuffle(shuffled_idx)

#     # Creating a list of start stop indicies.
#     if type(split_percent) is float or type(split_percent) is int:
#         split_ids = [0., int(split_percent*num_samples)]
#     else:
#         split_ids = [0.] + [int(x*num_samples) for x in split_percent]

#     split_ids = np.cumsum(split_ids, dtype = int)
#     if p_sum == 1:
#         split_ids[-1] = num_samples
#     else:
#         split_ids = np.concatenate((split_ids, [num_samples]))
    
    
#     # Create a list where each element in the list is another list containing the one split of all
#     # arrays in args.
#     # [[a1, b1], [a2, b2], ... [aN, bN]].
#     tmp_datasets = []
#     for i in range(len(split_ids)-1):
#         start = split_ids[i]
#         end = split_ids[i+1]
#         tmp_datasets.append([arr[shuffled_idx[start:end]] for arr in args])
    
#     # Create a list where each element in the list is another list containing all the splits of ONE 
#     # array in args.
#     # [[a1, a2, ..., aN], [b1, b2, ..., bN]]
#     new_datasets = []
#     for i in range(len(args)):
#         new_datasets.append([x[i] for x in tmp_datasets])
    
#     return new_datasets

# def CrossFoldValidate(x, y, model, n_fold, rng: np.random.Generator) -> float:
#     """
#     Performs n_fold cross validation with x and y.
#     x, y (np.ndarray): Feature and labels
#     model: The model to perform cross-fold validation on.
#     n_fold (int): the number of folds to use for cross fold validation.
#     rng: np.random.Generator

#     Return: The average accuracy across the 5 fold validation.
#     """
#     Xs, Ys = SplitData(x,y, split_percent=(1/n_fold,)*n_fold, rng=rng)
    
#     acc_log = []
#     loss_log = []
#     for i in range(n_fold):
#         model.fit(Xs[i], Ys[i])
#         test_x = np.vstack((*Xs[:i], *Xs[i+1:]))
#         test_y = np.concatenate((*Ys[:i], *Ys[i+1:]))
#         pred_y = model.predict(test_x)
#         acc_log.append(np.sum(test_y == pred_y)/test_y.shape[0])
#         loss_log.append(log_loss(test_y, pred_y))

#     return np.mean(acc_log), np.mean(loss_log)


# def SplitData_MP(*args, split_percent, rng = None, conn):
#     """
#     SplitData collectively splits the data (no replacement).
#     E.g. If given 2 arrays it will split the first and second array 
#     the same way.
#     *args (np.ndarray): arrays the should be split. The arrays should all have the same size first dimension.
#     split_percent (float or iterable of floats): How large each split should be.
#     conn: a multiprocesser child connector
#     """
#     # args = [arr.copy() for arr in args] 
#     num_samples = args[0].shape[0] # The number of samples

#     p_sum = np.sum(split_percent) # The sum of the probabilities.
#     if np.isclose(p_sum, 1, atol=1e-2):
#         p_sum = 1.

#     for arr in args:
#         assert type(arr) == np.ndarray, '*args must be of type np.ndarray!'

#     assert p_sum <= 1., 'Percentage of splits must sum to 1 or less.'
    
#     shuffled_idx = np.arange(num_samples) # Create a range of numbers to use for random sampling.
#     if rng:
#         rng.shuffle(shuffled_idx)
#     else:
#         np.random.shuffle(shuffled_idx)

#     # Creating a list of start stop indicies.
#     if type(split_percent) is float or type(split_percent) is int:
#         split_ids = [0., int(split_percent*num_samples)]
#     else:
#         split_ids = [0.] + [int(x*num_samples) for x in split_percent]

#     split_ids = np.cumsum(split_ids, dtype = int)
#     if p_sum == 1:
#         split_ids[-1] = num_samples
#     else:
#         split_ids = np.concatenate((split_ids, [num_samples]))
    
    
#     # Create a list where each element in the list is another list containing the one split of all
#     # arrays in args.
#     # [[a1, b1], [a2, b2], ... [aN, bN]].
#     tmp_datasets = []
#     for i in range(len(split_ids)-1):
#         start = split_ids[i]
#         end = split_ids[i+1]
#         tmp_datasets.append([arr[shuffled_idx[start:end]] for arr in args])
    
#     # Create a list where each element in the list is another list containing all the splits of ONE 
#     # array in args.
#     # [[a1, a2, ..., aN], [b1, b2, ..., bN]]
#     new_datasets = []
#     for i in range(len(args)):
#         new_datasets.append([x[i] for x in tmp_datasets])
    
#     conn.send(new_datasets)
#     conn.close()


# def TrainSVC_MP(x,y,model, conn):
#     """
#     Trains an SVC model. 
#     x, y: Features and labels
#     model: the SKlearn SVC model.
#     conn: A multiprocessing connector.

#     Output:
#     The trained model.
#     """
#     a = model.fit(x,y)
#     conn.send(a)
#     conn.close()