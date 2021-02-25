import numpy as np
from rpy2.robjects.packages import importr
from rpy2.robjects import r
from rpy2.robjects import FloatVector
import time
from config import N_KEEP
from numba import njit, prange
import math
import matplotlib.pyplot as plt
importr('BalancedSampling')


def project_one_chain(A, point):
    ### A is (N_constraints, N_particles) and point is (N_particles)
    return point - np.matmul(A.T,np.linalg.solve(np.matmul(A, A.T), np.matmul(A, point)))



def flight_phase(A, point):
    ### A is (N_particles, N_constraints) and point is (N_particles)
    signs = np.sign(point)
    A_signs = A.T.copy()
    A_signs[1:, :] = A_signs[1:, :]*signs
    point_signs = np.abs(point)*N_KEEP/np.sum(np.abs(point))
    A_signs *= point_signs
    A_signs_R = r['matrix'](FloatVector(np.array(list(A_signs.flatten()))), nrow=A_signs.T.shape[0])
    print("Flight phase:")
    res = r["flightphase"](FloatVector(point_signs), A_signs_R)
    print("Landing Phase:")
    land_point = r["landingphase"](FloatVector(point_signs), res, A_signs_R)
    return np.array([r for r in land_point]), signs


@njit()
def distance(x, y, Sigma=None):
    if Sigma is None:
        return np.sqrt(np.sum((x - y) ** 2))

    return np.sqrt(np.dot(x - y, np.linalg.solve(Sigma, x - y)))

@njit(parallel=True)
def compute_distance_matrix(x, y, Sigma=None):
    distance_matrix = np.zeros((x.shape[0], y.shape[0]))
    for i in prange(len(x)):
        for j in prange(len(y)):
            d_ij = distance(x[i, :], y[j, :], Sigma)
            distance_matrix[i, j] = d_ij

    return distance_matrix


def energyDistance(chain, selected, signs, omega, N_KEEP, full_chain_dist = 0):
    x = chain[np.abs(selected) == 1, :]
    y = chain[:]
    dist_mat = compute_distance_matrix(x,y)
    auto_dist_x = compute_distance_matrix(x, x)
    signs_matrix = np.matmul(signs[np.abs(selected) == 1][:, None], signs[np.abs(selected) == 1][None, :])
    return 2*(omega/(len(chain)*N_KEEP))* np.sum(signs[np.abs(selected) == 1]*np.sum(dist_mat, axis = 1)) - (omega/N_KEEP)**2*np.sum(
        signs_matrix*auto_dist_x) - full_chain_dist

@njit()
def log_star_second(z, n):
    if z >= 1/n:
        return -1/z**2

    return -n**2

@njit()
def log_star_first(z,n):
    if z >= 1/n:
        return 1/z

    return 2*n - n**2 * z


@njit()
def log_star(z,n):
    if z >= 1/n:
        return np.log(z)

    return np.log(1/n) - 1.5 + 2*n*z - (n*z)**2/2

log_star_second_vectorized = np.vectorize(log_star_second)
log_star_first_vectorized = np.vectorize(log_star_first)


@njit(parallel=True)
def compute_increments(constraints, lambd, n):
    ## Constraints is n_(sample, dimension_constraints)
    z = np.dot(constraints, lambd) + 1
    #coeff = np.sqrt(-log_star_second_vectorized(z, n))
    coeff = np.zeros((n, 1))
    for i in prange(n):
        coeff[i,:] = np.sqrt(-log_star_second(z[i], n))

    J = coeff*constraints
    mat_to_inv = np.dot(J.T, J)
    y = np.zeros(n)
    for i in prange(n):
        y[i] = log_star_first(z[i], n)/coeff[i, 0]

    #y = log_star_first_vectorized(z, n)/coeff
    print("Mat to invert")
    diag_mat = np.diag(np.diag(mat_to_inv))
    print(np.linalg.cond(np.dot(diag_mat, mat_to_inv)))
    print(np.linalg.det(np.dot(diag_mat, mat_to_inv)))
    _, v, _ = np.linalg.svd(np.dot(diag_mat, mat_to_inv))
    print(v)
    increment = np.linalg.solve(mat_to_inv, np.dot(J.T, y))
    return increment

@njit(parallel=True)
def compute_hessian(lambd, constraints, n):
    ## Constraints is n_(sample, dimension_constraints)
    z = np.dot(constraints, lambd) + 1
    #coeff = np.sqrt(-log_star_second_vectorized(z, n))
    coeff = np.zeros((n, 1))
    for i in prange(n):
        coeff[i,:] = np.sqrt(-log_star_second(z[i], n))

    J = coeff*constraints
    hessian = np.dot(J.T, J)
    return hessian

@njit(parallel=True)
def compute_gradient(lambd, constraints, n):
    ## Constraints is n_(sample, dimension_constraints)
    z = np.dot(constraints, lambd) + 1
    #coeff = np.sqrt(-log_star_second_vectorized(z, n))
    coeff = np.zeros((n, 1))
    for i in prange(n):
        coeff[i,:] = np.sqrt(-log_star_second(z[i], n))

    J = coeff*constraints
    y = np.zeros(n)
    for i in prange(n):
        y[i] = log_star_first(z[i], n)/coeff[i, 0]

    grad = np.dot(J.T, y)
    return grad


#@njit(parallel=True)
#def func_to_minimize(lambd, constraints, n):


def newton_raphson(lambda_init, constraints):
    n = len(constraints)
    lambd = lambda_init
    for i in range(100):
        print(i)
        weights = (1 / (np.dot(constraints, lambd) + 1)) * (1 / len(constraints))
        print(np.mean(weights < 0))
        print("Lambda norrm:", np.sqrt(np.sum(lambd**2)))
        inc = compute_increments(constraints, lambd, n)
        lambd += inc
        print("\n")

    return lambd


@njit(parallel=True)
def compute_distance_matrix(x, y, Sigma=None):
    distance_matrix = np.zeros((x.shape[0], y.shape[0]))
    for i in prange(len(x)):
        for j in prange(len(y)):
            d_ij = distance(x[i, :], y[j, :], Sigma)
            distance_matrix[i, j] = d_ij

    return distance_matrix



@njit()
def transform_to_unif(x, chol_sigma_approx, mu_approx):
    standard_normal = np.linalg.solve(chol_sigma_approx, x - mu_approx)
    #return stats.norm.cdf(standard_normal)
    res = np.zeros(len(standard_normal))
    for k in prange(len(standard_normal)):
        res[k] = (1/2)*(1 + math.erf(standard_normal[k]/np.sqrt(2)))

    return res


@njit(parallel=True)
def discrepency(chain1, chain2, chol_sigma_approx, mu_approx, n_max=100000, weights = None):
    n_chain1 = len(chain1)
    n_chain2 = len(chain2)
    d = chain1.shape[1]
    all_us_chain1 = np.zeros((n_chain1, chain1.shape[1]))
    for i in prange(n_chain1):
        u = transform_to_unif(chain1[i, :], chol_sigma_approx, mu_approx)
        all_us_chain1[i, :] = u

    all_us_chain2 = np.zeros((n_chain2, chain2.shape[1]))
    for i in prange(n_chain2):
        u = transform_to_unif(chain2[i, :], chol_sigma_approx, mu_approx)
        all_us_chain2[i, :] = u

    all_unifs = np.random.uniform(0, 1, size=(n_max, d))
    all_discrepancies = np.zeros(n_max)
    for i in prange(n_max):
        bools1 = (all_us_chain1 < all_unifs[i, :])
        res1 = np.ones(len(bools1))

        bools2 = (all_us_chain2 < all_unifs[i, :])
        res2 = np.ones(len(bools2))
        for m in range(bools1.shape[1]):
            res1 *= bools1[:, m]
            res2 *= bools2[:, m]

        if weights is None:
            all_discrepancies[i] = np.abs(
                np.mean(res1) - np.mean(res2))
        else:
            all_discrepancies[i] = np.abs(
                np.sum(res1*weights) - np.mean(
                    res2))

    return np.max(all_discrepancies)
