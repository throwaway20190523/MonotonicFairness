import numpy as np
import tensorflow as tf

"""
 Equation numbers are from: 
 Zemel, Rich, et al. "Learning fair representations." International Conference on Machine Learning. 2013.
 https://www.cs.toronto.edu/~toni/Papers/icml-final.pdf
"""

# Eq 13
def accuracy_np(Y, Yhat):
    yAcc = 1 - np.abs(Y - Yhat).mean()
    return yAcc


# Eq 13
def accuracy_tf(Y, Yhat):
    yAcc = 1 - tf.reduce_mean(tf.abs(Y - Yhat))
    return yAcc


# Eq 14
def discrimination_np(A, Yhat):
    sums = (Yhat.transpose().dot(A) / A.sum(0))[0, :]
    return np.abs(sums[1] - sums[0])


# Eq 14
def discrimination_tf(A, Yhat):
    sums = tf.matmul(tf.transpose(Yhat), A) / tf.reduce_sum(A, axis=0)
    return np.abs(sums[1] - sums[0])


def k_nearest_neighbors_sp(
    X,
    k_nearest = 6,
    self_normalize = True,
    sigmas = None,
):
    n, x_dim = X.shape
    if sigmas is None:
        sigmas = 1.
    tups = [
        [(i, j, -1./k_nearest if self_normalize else 1.) 
         for j 
         in ((((X - x) ** 2) / sigmas).sum(1)).argsort()[1:k_nearest+1]
         ] + 
        ([(i, i, 1.)] if self_normalize else []) 
        for i, x 
        in enumerate(X)
    ]
    row, col, val = zip(*[tup for tup_list in tups for tup in tup_list])
    from scipy.sparse import csr_matrix 
    return csr_matrix(
        (val, (row, col)),
        shape=[n, n]
    )


# Eq 15
def consistency_np(Yhat, nearest_neighbors):
    # assumes 
    #  Yhat is [n, 1]
    #  nearest_neighbors (sparse OK) is [n, n], where 
    #    [i, i] = 1.
    #    [i, j] = -1./k iff. x_j is one of x_i's closest k neighbors
    return 1. - np.abs(nearest_neighbors.dot(Yhat)).mean()


def identify_monotonic_pairs(
    x1,
    x2 = None,
    monotonicity = None,
):
    self_compare = (x2 is None)
    if x2 is None:
        x2 = x1
    
    n1, x_dim = x1.shape
    n2 = x2.shape[0]
    
    if monotonicity is None:
        monotonicity = [1] * x_dim
    mono = np.array(monotonicity)
    x1_mod = x1 * mono
    x2_mod = x2 * mono
    
    i = 0
    j = 1
    monotonic_pairs = np.array([
        [i, j]
        for i in range(n1)
        for j in range(n2)
        if (x1_mod[i, :] > x2_mod[j, :]).max()
         * (x1_mod[i, :] >= x2_mod[j, :]).min()
         * ((x1[i, :] * (1. - np.abs(mono))) == (x2[j, :] * (1. - np.abs(mono)))).min()
    ])
    monotonic_dict = {i:[] for i in range(n1)}
    for i, j in monotonic_pairs:
        monotonic_dict[i].append(j)
    
    monotonic_dict
    return monotonic_pairs, monotonic_dict


def resentment_pairwise(
    yhat1,
    monotonic_pairs,
    yhat2 = None,
):
    if yhat2 is None:
        yhat2 = yhat1
    
    return (
        yhat1[monotonic_pairs[:, 0]] 
        < 
        yhat2[monotonic_pairs[:, 1]]
    ).mean()


def resentment_individual(
    yhat1,
    monotonic_dict,
    yhat2 = None,
):
    if yhat2 is None:
        yhat2 = yhat1

    individuals_with_resentment = np.zeros([yhat1.shape[0]])
    for i in monotonic_dict.keys():
        for j in monotonic_dict[i]:
            if yhat1[i] < yhat2[j]:
                individuals_with_resentment[i] = True
                break
    return individuals_with_resentment.mean()


def lipschitz_sample_estimate(
    X_np,
    Y_hat_np,
    X_means = None,
    X_sds = None,
):
    n, x_dim = X_np.shape
    if X_means is None and X_sds is None:
        X_means = np.zeros([x_dim])
        X_sds = np.ones([x_dim])
        
    X = (X_np - X_means) / X_sds
    Y = Y_hat_np
    
    lower_bound = (np.abs(
        np.tile(Y.reshape([n, 1]), [1, n]) - 
        np.tile(Y.reshape([1, n]), [n, 1]) 
    ) / np.sqrt(np.square( 
        np.tile(X.reshape([n, 1, x_dim]), [1, n, 1]) - 
        np.tile(X.reshape([1, n, x_dim]), [n, 1, 1]) +
        1e-8
    ).sum(2))).max()
    return lower_bound
