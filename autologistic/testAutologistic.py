import statsmodels.api as sm
import numpy as np
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import utils_autologistic
import argparse
from sklearn.linear_model import LogisticRegression
import time
from utils import get_weights, cube_method, get_beta



parser = argparse.ArgumentParser(description="run Noise contrastive estimation")
parser.add_argument("data_path", help="path of observed data")
parser.add_argument("output_path", help="path of the output weigths")
parser.add_argument("n_experiments", help="number of experiments to run", type=int)
parser.add_argument("N_gibbs_step", help="number of gibbs step", type=int)



arguments = parser.parse_args()

data_path = arguments.data_path
n_experiments = arguments.n_experiments
n_gibbs = arguments.N_gibbs_step
output_path = arguments.output_path




d = np.load(data_path, allow_pickle=True)
d = d.item()
y_obs = d["data"]
neighbour_matrix = d["neighbours"]
print(neighbour_matrix.shape)
n_points = len(y_obs)
alpha_nce = d["alpha"]
beta_nce = d["beta"]
all_means = []
all_means_cv = []
for k in range(n_experiments):
    print("Iteration:", k)
    start = time.time()
    start_clock = time.clock()
    y = np.random.binomial(1, 0.5, n_points) * 2 - 1
    h_y = utils_autologistic.run_gibbs_wrapper(y.copy(), alpha_nce, beta_nce, neighbour_matrix, n_iter=n_gibbs, history=True)

    constraints = utils_autologistic.create_constraints(h_y, alpha_nce, beta_nce, neighbour_matrix)
    X = sm.add_constant(constraints)
    reg = sm.OLS(h_y[:, 0], X)
    results = reg.fit()
    intercept = results.params[0]
    estimated_mean = np.mean(h_y[:, 0])
    all_means.append(estimated_mean)
    all_means_cv.append(intercept)


all_means = np.array(all_means)
all_means_cv = np.array(all_means_cv)

d = {"means":all_means, "means_cv":all_means_cv}
np.save(output_path, d, allow_pickle=True)

print("Estimates:")
print(np.mean(all_means))
print(np.mean(all_means_cv))
print("Variances estimates:")
print(np.var(all_means))
print(np.var(all_means_cv))
