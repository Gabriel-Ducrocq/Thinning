import numpy as np
from numba import njit, prange
from scipy import stats

@njit(parallel=True)
def fill(mat):
    for i in prange(len(mat)):
        mat[i, :] = np.random.binomial(1, 0.5, len(mat))

    np.fill_diagonal(mat, 0)
    return mat


@njit()
def compute_features(y, neighbours):
    alpha_feature = np.sum(y)
    beta_feats = np.zeros(len(y))
    for i in prange(len(y)):
        beta_feats[i] = np.sum(y[neighbours[i, :] == 1])

    beta_feature = np.sum(beta_feats*y)/2
    return alpha_feature, beta_feature

@njit(parallel=True)
def compute_dataset_regression(all_y, neighbours):
    features = np.zeros((len(all_y), 2))
    for i in prange(len(all_y)):
        a_feat, b_feat = compute_features(all_y[i, :], neighbours)
        features[i, 0] = a_feat
        features[i, 1] = b_feat

    return features


def gibbs_iteration(y, alpha, beta, i, neighbours):
    exponent = alpha + (beta/2)*np.sum(y[neighbours[i, :] == 1])
    proba = np.exp(exponent)/(np.exp(exponent) + np.exp(-exponent))
    if np.random.uniform() < proba:
        y[i] = 1
    else:
        y[i] = -1

    return y


def run_gibbs(y, alpha, beta, neighbours, n_iter=1000, history =True):
    h_y = []
    conditional_indexes = stats.randint.rvs(low=0, high=len(y), size=n_iter)
    for index in conditional_indexes:
        y = gibbs_iteration(y.copy(), alpha, beta, index ,neighbours)
        h_y.append(y)

    if history:
        return np.array(h_y)

    return y


@njit(parallel=True)
def create_constraints(points: list, alpha: float, beta:float, neighbours: list):
    constraints = np.zeros((len(points), points.shape[1]))
    for i in prange(len(points)):
        y = points[i, :]
        for j in range(points.shape[1]):
            exponent = alpha + (beta / 2) * np.sum(y[neighbours[j, :] == 1])
            proba = np.exp(exponent) / (np.exp(exponent) + np.exp(-exponent))
            constraints[i, j] = y[j] - 2*proba - 1

    return constraints





