import numpy as np
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import utils_autologistic
import argparse
from sklearn.linear_model import LogisticRegression
import time
from utils import get_weights, cube_method



parser = argparse.ArgumentParser(description="run Noise contrastive estimation")
parser.add_argument("data_path", help="path of observed data")
parser.add_argument("output_path", help="path of the output weigths")
parser.add_argument("n_experiments", help="number of experiments to run", type=int)
parser.add_argument("N_gibbs_step", help="number of gibbs step", type=int)
parser.add_argument("alpha_nce", help="alpha", type=float)
parser.add_argument("beta_nce", help="beta", type=float)
parser.add_argument("cube", help="whether to use cube method or not", type=int)
parser.add_argument("thinning", help="whether to use usual thinning or not", type=int)
parser.add_argument("N_KEEP", help="Number of points to keep", type=int)



arguments = parser.parse_args()

data_path = arguments.data_path
n_experiments = arguments.n_experiments
alpha_nce = arguments.alpha_nce
beta_nce = arguments.beta_nce
n_gibbs = arguments.N_gibbs_step
output_path = arguments.output_path
useCube = arguments.cube
useThinning = arguments.thinning
N_KEEP = arguments.N_KEEP




d = np.load(data_path, allow_pickle=True)
d = d.item()
y_obs = d["data"]
neighbour_matrix = d["neighbours"]
n_points = len(y_obs)


all_coefs = []
all_intercept = []
all_signs = []
all_selected = []
all_ED = []
all_cpu_times = []
start = time.time()

for k in range(n_experiments):
    print("Iteration:", k)
    start = time.time()
    start_clock = time.clock()
    y = np.random.binomial(1, 0.5, n_points) * 2 - 1
    h_y = utils_autologistic.run_gibbs_wrapper(y.copy(), alpha_nce, beta_nce, neighbour_matrix, n_iter=n_gibbs, history=True)

    if useCube == 1 and useThinning == 0:
        constraints = utils_autologistic.create_constraints(h_y, alpha_nce, beta_nce, neighbour_matrix)
        weights = get_weights(constraints)
        omega = np.sum(np.abs(weights))
        print("SUM OF PROJECTED POINTS:", np.sum(weights))
        print("Constraints respected ?", np.dot(constraints.T, weights))
        constraints = np.hstack([np.ones((constraints.shape[0], 1)), constraints])
        selected, signs = cube_method(constraints, weights, N_KEEP)
        print("Number of points kept:", np.sum(np.abs(selected)))
        print("Sum of points:", np.sum(signs))
        print("Proportion of positive weights:",len(signs[signs == 1])/len(signs))
        points_weights = signs[np.abs(selected) == 1]*omega/N_KEEP
        h_y = h_y[np.abs(selected) == 1, :]
    else:
        points_weights = np.ones(len(h_y))

    feature_matrix_true = utils_autologistic.compute_dataset_regression(np.array([y_obs]), neighbour_matrix)
    feature_matrix_false = utils_autologistic.compute_dataset_regression(h_y, neighbour_matrix)

    dataset = np.vstack([feature_matrix_true, feature_matrix_false])
    label = np.zeros(len(dataset))
    label[:1] = 1

    points_weights = np.concatenate([np.ones(1), points_weights])
    regressor = LogisticRegression(penalty='none', solver="lbfgs", fit_intercept=True)
    regressor.fit(dataset, label, sample_weight=points_weights)
    coefs = regressor.coef_
    intercept = regressor.intercept_

    all_coefs.append(coefs)
    all_intercept.append(intercept)
    duration_clock = time.clock() - start_clock
    all_cpu_times.append(duration_clock)
    print("Iteration time:", time.time() - start)
    print("coefs:", coefs + np.array([alpha_nce, beta_nce]))

all_coefs = np.array(all_coefs)
all_intercept = np.array(all_intercept)



d = {"coefs":all_coefs, "intercepts":all_intercept, "alpha_nce": alpha_nce,
     "beta_nce":beta_nce, "neighbour":neighbour_matrix, "input_data":data_path, "cube":useCube, "thinning":useThinning,
     "N_KEEP":N_KEEP, "all_times":all_cpu_times}

np.save(output_path, d, allow_pickle=True)
"""
alpha_nce = 0.01
beta_nce = 0.05
d = np.load("test.npy", allow_pickle=True)
d = d.item()

coefs = d["coefs"]

print(coefs.shape)


fig, ax = plt.subplots(1, 2)
ax[0].hist(coefs[:, 0, 0] + alpha_nce, alpha=0.5, bins = 15)
ax[0].set_title("alpha")
ax[0].axvline(x=alpha, color="red")

ax[1].hist(coefs[:, 0, 1] + beta_nce, alpha=0.5, bins = 15)
ax[1].set_title("beta")
ax[1].axvline(x=beta, color="red")
plt.show()
"""
