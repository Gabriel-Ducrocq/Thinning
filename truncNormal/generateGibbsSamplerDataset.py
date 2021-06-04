from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np
import utils_truncated
import sys
from scipy.stats import truncnorm, norm
import argparse
import statsmodels.api as sm
import os, sys, time
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from utils import get_weights, cube_method


parser = argparse.ArgumentParser(description="runs NCE")
parser.add_argument("datapath", help="data path")
parser.add_argument("output_path", help="path of the output weigths")
parser.add_argument("N_KEEP", help="Number of points to keep", type=int)
parser.add_argument("n_iter_gibbs", help="Number of gibbs step", type=int)
parser.add_argument("n_experiments", help="number of experiments", type=int)
parser.add_argument("cube", help="Whether we apply a cube method", type = int)
parser.add_argument("thin", help="Whether we apply thinning", type = int)
parser.add_argument("reg", help="Whether we use the regression estiator of Art Owen", type=int)

arguments = parser.parse_args()
data = np.load(arguments.datapath, allow_pickle=True)
data = data.item()
mu = data["mu"]
sigma = data["sigma"]
M = arguments.N_KEEP
output_path = arguments.output_path
n_iter = arguments.n_iter_gibbs
n_experiments = arguments.n_experiments
use_cube = arguments.cube
use_thin = arguments.thin
use_reg = arguments.reg
#mu = np.random.normal(size = 10)
#m = np.random.normal(size = (10, 10))
#sigma = np.dot(m, m.T)

coefficients = []
intercepts = []
all_means = []
all_times = []
h_y = data
all_cube = []
for k in range(n_experiments):
    start = time.time()
    h_x = []
    x = np.ones(len(mu))*2
    print(k)
    for i in range(n_iter):
        l = np.random.randint(0, len(mu))
        x[l] = utils_truncated.gibbs_step(l, x.copy(), mu, sigma)
        h_x.append(x.copy())

    h_x = np.array(h_x)
    print("Regular estimator")
    print(np.mean(h_x, axis = 0))

    if use_cube == 1 and use_thin == 0:
        print("Control variates computing")
        control_variates = utils_truncated.construct_cv(h_x.copy(), mu, sigma)
        print("Weights computing")
        weights = get_weights(control_variates)
        print("Stacking")
        control_variates = np.hstack((np.ones((len(control_variates), 1)),control_variates))
        print(np.dot(control_variates.T, weights))
        print("Cube method")
        points_keep, signs = cube_method(control_variates, weights, M)
        print(np.sum(points_keep))
        print("Porportion positive:", np.mean(weights > 0))
        h_x = h_x[points_keep == 1, :]
        print(h_x.shape)
        omega = np.sum(np.abs(weights))
        estimator = (omega/M)*np.sum(signs[points_keep == 1, None]*h_x, axis = 0)
        print("Cube estimator:")
        print(estimator)
        all_cube.append(estimator)

    if use_thin == 1 and use_cube == 0:
        indexes = np.linspace(0, len(h_x)-1, M, dtype = int)
        h_x = h_x[indexes, :]
        print("Shape new h_x:", h_x.shape)
        estimator = np.mean(h_x, axis = 0)
        print("Thin estimator:")
        print(estimator)
        all_cube.append(estimator)

    if use_thin == 0 and use_cube == 0 and use_reg == 0:
        all_cube.append(np.mean(h_x, axis = 0))

    if use_thin == 0 and use_cube == 0 and use_reg == 1:
        control_variates = utils_truncated.construct_cv(h_x.copy(), mu, sigma)
        linReg = LinearRegression()
        linReg.fit(control_variates, h_x)
        all_cube.append(linReg.intercept_)

d = {"all_estimator": np.array(all_cube), "sigma":sigma, "mu":mu, "N_KEEP":M, "gibbs_iter":n_iter}
np.save(output_path, d, allow_pickle=True)
