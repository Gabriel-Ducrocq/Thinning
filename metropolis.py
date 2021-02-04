import numpy as np
from scipy import stats
from config import dimension, n_particles, n_chains

m = np.random.uniform(low = -1, high = 1, size = (dimension, dimension))
sigma_target = np.matmul(m.T, m)
mu_target = np.ones(dimension)*10
sigma_proposal = 0.6*sigma_target


def propose(x, sigma_proposal, n_chains):
    return x + stats.multivariate_normal.rvs(cov=sigma_proposal, size=n_chains)


def compute_log_proposal(x_new, x_old, precision_proposal):
    diff = x_new - x_old
    return -(1 / 2) * np.sum(np.matmul(diff, precision_proposal) * diff, axis=1)


def grad_log_density(x, precision_target):
    return -np.matmul(x - mu_target, precision_target)


def compute_log_density(x, precision_target):
    return -(1 / 2) * np.sum(np.matmul(x - mu_target, precision_target) * (x - mu_target), axis=1)


def compute_log_ratio(x_new, x_old, precision_target, precision_proposal):
    first_part = compute_log_density(x_new, precision_target) - compute_log_density(x_old, precision_target)
    second_part = compute_log_proposal(x_old, x_new, precision_proposal) - compute_log_proposal(x_new,
                                                                                                x_old,
                                                                                                precision_proposal)
    log_ratio = first_part + second_part
    return log_ratio


def run_metropolis(sigma_target, sigma_proposal, starting_points, n_iterations=100000, n_chains=15):
    ##Rows are parallel chains, Ncolumns is dimension of state space
    precision_target = np.linalg.inv(sigma_target)
    precision_proposal = np.linalg.inv(sigma_proposal)

    h_x = []
    h_grad = []
    x = starting_points
    all_acceptions = np.zeros((n_chains, 1))
    for i in range(n_iterations):
        h_x.append(x)
        h_grad.append(grad_log_density(x, precision_target))
        new_x = propose(x, sigma_proposal, n_chains)
        log_ratio = compute_log_ratio(new_x, x, precision_target, precision_proposal)
        log_u = np.log(np.random.uniform(size=n_chains))
        acceptions = (log_u < log_ratio)[:, None]
        x = x * (1 - acceptions) + new_x * acceptions
        all_acceptions += acceptions

    h_x = np.array(h_x)
    h_grad = np.array(h_grad)
    print("Acceptance rates:", all_acceptions / n_iterations)
    return h_x, all_acceptions, h_grad

starting_point = np.ones(shape=(n_chains, dimension))*50
h_x, acceptions, h_grad = run_metropolis(sigma_target, sigma_proposal, starting_point, n_chains=n_chains, n_iterations=n_particles)

d = {"h_x":h_x[:, 0, :], "acceptions":acceptions, "h_grad":h_grad[:, 0, :]}

np.save("chain.npy", d, allow_pickle=True)

