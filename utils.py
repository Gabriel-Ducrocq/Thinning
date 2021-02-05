import numpy as np
from rpy2.robjects.packages import importr
from rpy2.robjects import r
from rpy2.robjects import FloatVector
import time
from config import N_KEEP
from numba import njit, prange
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


def log_star_second(z, n):
    if z >= 1/n:
        return -1/z**2

    return -n**2


def log_star_first(z,n):
    if z >= 1/n:
        return 1/z

    return 2*n - n**2 * z

log_star_second_vectorized = np.vectorize(log_star_second)
log_star_first_vectorized = np.vectorize(log_star_first)


def compute_increments(constraints, lambd, n):
    ## Constraints is n_(sample, dimension_constraints)
    z = np.dot(constraints, lambd) + 1
    print("z:", z)
    print("total:", -log_star_second_vectorized(z, n))
    coeff = np.sqrt(-log_star_second_vectorized(z, n))
    J = coeff[:, None]*constraints
    mat_to_inv = np.dot(J.T, J)
    y = log_star_first_vectorized(z, n)/coeff
    increment = np.linalg.solve(mat_to_inv, np.dot(J.T, y))
    return increment

def newton_raphson(lambda_init, constraints):
    n = len(constraints)
    lambd = lambda_init
    for i in range(20):
        print(i)
        weights = (1 / (np.dot(constraints, lambd) + 1)) * (1 / len(constraints))
        print(np.mean(weights < 0))
        inc = compute_increments(constraints, lambd, n)
        lambd += inc
        print("\n")

    return lambd





#print(log_star_second_vectorized(1.01383924, 2000000))