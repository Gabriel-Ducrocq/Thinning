import numpy as np
from scipy import stats
from config import dimension, n_particles, n_chains
import argparse

parser = argparse.ArgumentParser(description="Whether it is a tuning run")
parser.add_argument("chain_path", help="path of the chain to use as preliminary")
parser.add_argument("tuning", help="whether it is a preliminary run or not", type=int)
parser.add_argument("tau", help ="step size", type=float)
parser.add_argument("output_path", help="path of the output")
arguments = parser.parse_args()

m = np.random.uniform(low = -1, high = 1, size = (dimension, dimension))
sigma_target = np.load(arguments.chain_path, allow_pickle=True)
sigma_target = sigma_target.item()
sigma_target = sigma_target["sigma_target"]
#sigma_target = np.matmul(m.T, m)
mu_target = np.ones(dimension)*10

if arguments.tuning == 1:
    sigma_proposal = np.eye(len(mu_target))
else:
    tuning_run = np.load(arguments.chain_path, allow_pickle=True)
    tuning_run = tuning_run.item()
    h_x = tuning_run["h_x"]
    sigma_proposal = np.cov(h_x.T)

tau =arguments.tau


def propose(x,tau, grad_log, sigma_proposal, chol_sigma_proposal):
    return x + tau*np.matmul(sigma_proposal, grad_log) + np.sqrt(2*tau)*np.matmul(chol_sigma_proposal, np.random.normal(size = len(x)))


def compute_log_proposal(x_new, x_old, tau, grad_log, sigma_proposal, precision_proposal):
    diff = x_new - (x_old + tau* np.matmul(sigma_proposal, grad_log))
    return -(1 / 2) * np.sum(np.matmul(diff, precision_proposal/(2*tau)) * diff)


def grad_log_density(x, precision_target):
    return -np.matmul(x - mu_target, precision_target)


def compute_log_density(x, precision_target):
    return -(1 / 2) * np.sum(np.matmul(x - mu_target, precision_target) * (x - mu_target))


def compute_log_ratio(x_new, x_old, tau, grad_log_old, grad_log_new, precision_target, precision_proposal, sigma_target, sigma_proposal, chol_sigma_proposal):
    first_part = compute_log_density(x_new, precision_target) - compute_log_density(x_old,precision_target)
    second_part = compute_log_proposal(x_old, x_new, tau, grad_log_new,sigma_proposal, precision_proposal) - compute_log_proposal(x_new,
                                                                                                x_old,tau, grad_log_old,sigma_proposal,
                                                                                                precision_proposal)
    log_ratio = first_part + second_part
    return log_ratio


def run_metropolis(sigma_target, sigma_proposal, starting_points, n_iterations=1000000):
    ##Rows are parallel chains, Ncolumns is dimension of state space
    precision_target = np.linalg.inv(sigma_target)
    precision_proposal = np.linalg.inv(sigma_proposal)
    chol_sigma_proposal = np.linalg.cholesky(sigma_proposal)

    h_x = []
    h_grad = []
    x = starting_points
    all_acceptions = 0
    grad_log = grad_log_density(x,precision_target)
    for i in range(n_iterations):
        if i%100== 0:
            print(i)

        h_x.append(x)
        h_grad.append(grad_log)
        new_x = propose(x, tau,grad_log, sigma_proposal, chol_sigma_proposal)
        grad_log_new = grad_log_density(new_x, precision_target)
        log_ratio = compute_log_ratio(new_x, x,tau, grad_log,grad_log_new,  precision_target, precision_proposal, sigma_target, sigma_proposal, chol_sigma_proposal)
        log_u = np.log(np.random.uniform())
        if log_u < log_ratio:
            x = new_x
            grad_log = grad_log_new
            all_acceptions += 1

    h_x = np.array(h_x)
    h_grad = np.array(h_grad)
    print("Acceptance rates:", all_acceptions / n_iterations)
    return h_x, all_acceptions, h_grad

starting_point = np.ones(dimension)*50
h_x, acceptions, h_grad = run_metropolis(sigma_target, sigma_proposal, starting_point, n_iterations=n_particles)

d = {"h_x":h_x, "acceptions":acceptions, "h_grad":h_grad, "sigma_target":sigma_target, "sigma_proposal":sigma_proposal, "mu_target":mu_target, "tau":tau}

np.save(arguments.output_path, d, allow_pickle=True)

