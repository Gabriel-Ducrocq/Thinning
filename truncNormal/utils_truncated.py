from scipy.stats import truncnorm
import numpy as np
from numba import njit, prange

def gibbs_step(i, x, mu, sigma):
    cov_i = sigma[i, i]
    complement = [l for l in range(len(mu)) if l != i]
    cov_complement = sigma[complement][:, complement]
    sigma_cross = sigma[i, complement]
    mean_cond = mu[i] + np.dot(sigma_cross, np.linalg.solve(cov_complement, x[complement]-mu[complement]))
    sigma_cond = cov_i - np.dot(sigma_cross,np.linalg.solve(cov_complement, sigma_cross.T))
    a = (0 - mean_cond)/np.sqrt(sigma_cond)
    return truncnorm.rvs(a, np.inf, mean_cond, np.sqrt(sigma_cond))


def get_mean(i, x, mu, sigma):
    cov_i = sigma[i, i]
    complement = [l for l in range(len(mu)) if l != i]
    cov_complement = sigma[complement][:, complement]
    sigma_cross = sigma[i, complement]
    mean_cond = mu[i] + np.dot(sigma_cross, np.linalg.solve(cov_complement, x[complement]-mu[complement]))
    sigma_cond = cov_i - np.dot(sigma_cross,np.linalg.solve(cov_complement, sigma_cross.T))
    return truncnorm(-mean_cond/np.sqrt(sigma_cond), np.inf, mean_cond, np.sqrt(sigma_cond)).mean()

def get_control_variates(x, mu, sigma):
    cv = np.zeros(len(mu))
    for i in range(len(mu)):
        cv[i] = get_mean(i, x, mu, sigma)

    return cv

def construct_cv(h_x, mu, sigma):
    control_var = np.zeros(h_x.shape)
    for l in prange(len(h_x)):
        control_var[l, :] = get_control_variates(h_x[l, :], mu, sigma)

    control_var = h_x - control_var
    return control_var

def compute_regressors(x, triu_indices, dim):
    all_reg = np.zeros(dim)
    m=np.outer(x, x)
    np.fill_diagonal(m, np.diag(m)/2)
    reg_cov = m[triu_indices]
    reg_mean = x[:]
    all_reg[:len(x)] = reg_mean
    all_reg[len(x):] = -reg_cov
    return all_reg

def compute_all_regressors(h_x, triu_indices):
    dim_regressors = int((h_x.shape[1] + 1)*h_x.shape[1]/2 + h_x.shape[1])
    regressors = np.zeros((h_x.shape[0], dim_regressors))
    for i in prange(len(h_x)):
        regressors[i, :] = compute_regressors(h_x[i, :], triu_indices, dim_regressors)

    return regressors
