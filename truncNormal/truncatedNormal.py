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

arguments = parser.parse_args()
data = np.load(arguments.datapath, allow_pickle=True)
data = data.item()
mu_true = data["mu"]
sigma_true = data["Sigma"]
regressors_true = data["regressors"]
data = data["points"]
M = arguments.N_KEEP
output_path = arguments.output_path
n_iter = arguments.n_iter_gibbs
n_experiments = arguments.n_experiments
use_cube = arguments.cube
use_thin = arguments.thin
#mu = np.zeros(len(mu_true))+1
#sigma = np.eye(len(mu_true))*0.5
mu = np.zeros(len(mu_true))
sigma = np.array([[1.54778467, 0.77358515, 0.53010819],
       [0.77358515, 2.4881138 , 0.46121046],
       [0.53010819, 0.46121046, 1.79801314]])

coefficients = []
intercepts = []
all_means = []
all_times = []
#h_true = truncnorm.rvs(-mu_true/np.sqrt(sigma_true), np.inf, mu_true, np.sqrt(sigma_true), size = 100000)
#regressors_true = np.vstack([h_true, -(1/2)*h_true**2]).T
#h_y = np.zeros((1000, 3))
#h_y[:, 0] = truncnorm.rvs(-mu_true[0]/np.sqrt(sigma_true[0, 0]), np.inf, mu_true[0], np.sqrt(sigma_true[0, 0]), size=len(h_y))
#h_y[:, 1] = truncnorm.rvs(-mu_true[1]/np.sqrt(sigma_true[1, 1]), np.inf, mu_true[1], np.sqrt(sigma_true[1, 1]), size=len(h_y))
#h_y[:, 2] = truncnorm.rvs(-mu_true[2]/np.sqrt(sigma_true[2,2]), np.inf, mu_true[2], np.sqrt(sigma_true[2, 2]), size=len(h_y))
h_y = data
for k in range(n_experiments):
    start = time.time()
    h_x = []
    x = np.ones(len(mu))*2
    print(k)
    for i in range(n_iter):
        l = np.random.randint(0, len(mu))
        x[l] = utils_truncated.gibbs_step(l, x.copy(), mu, sigma)
        h_x.append(x.copy())
    #h_x = truncnorm.rvs(-mu/np.sqrt(sigma), np.inf, mu, np.sqrt(sigma), size = 100000)
    h_x = np.array(h_x)
    #regressors_false = np.hstack([h_x, -(1/2)*h_x**2])
    #regressors_true = np.hstack([h_y, -(1/2)*h_y**2])
    #print(regressors_false)
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
        #kept_signs = signs[points_keep==1]
        #print("Check",np.mean(kept_signs* np.sum(np.abs(weights))/M> 0))

    if use_thin == 1 and use_cube == 0:
        indexes = np.linspace(0, len(h_x)-1, M, dtype = int)
        h_x = h_x[indexes, :]
        print("Shape new h_x:", h_x.shape)
    triu_indices = np.triu_indices(h_x.shape[1])
    regressors_false = utils_truncated.compute_all_regressors(h_x, triu_indices)
    #regressors_false= np.hstack([h_x[:, 0][:, None], h_x[:, 1][:, None],-(1/2)*h_x[:, 0][:, None]**2, -(1/2)*h_x[:, 1][:, None]**2])
    #regressors_true= np.hstack([h_y[:, 0][:, None], h_y[:, 1][:, None],-(1/2)*h_y[:, 0][:, None]**2, -(1/2)*h_y[:, 1][:, None]**2])
    regressors_true = utils_truncated.compute_all_regressors(h_y, triu_indices)
    labels = np.zeros(len(regressors_false)+len(regressors_true))
    labels[:len(regressors_true)] = 1
    #print(regressor_true.shape)
    #print(regressor_false.shape)
    all_points = np.vstack((regressors_true, regressors_false))
    logit = LogisticRegression(solver="newton-cg",penalty="none", fit_intercept = True, max_iter = 10000)
    logit.fit(all_points, labels)
    print(logit.coef_)
    coefficients.append(logit.coef_[0, :])
    intercepts.append(logit.intercept_)
    #print(np.linalg.solve(sigma_true, mu_true))
    #X = sm.add_constant(control_variates)
    #reg = sm.OLS(h_x[:, 0], X)
    #results = reg.fit()
    #intercept = results.params[0]
    #intercepts.append(intercept)
    #print(intercept)
    #print(np.mean(h_x[:, 0]))
    #all_means.append(np.mean(h_x[:, 0]))
    end = time.time()
    print("Time",end - start)
    all_times.append(end-start)

coefficients = np.array(coefficients)
intercepts = np.array(intercepts)
all_times = np.array(all_times)
r_true = np.linalg.solve(sigma_true, mu_true)
q_true = np.linalg.inv(sigma_true)
print("R true", r_true)
print("Intercept")
print("Shape:",coefficients.shape)
print("Mean", np.mean(coefficients, axis =0))
print("Std", np.std(coefficients, axis = 0))
results = {"all_times":all_times, "gibbs_iter":n_iter, "N_KEEP":M ,"use_cube":use_cube, "use_thin":use_thin,"mu_true":mu_true, "sigma_true":sigma_true, "mu_false":mu, "sigma_false":sigma, "estim_coeffs":coefficients, "intercepts":intercepts, "input_path":arguments.datapath}
np.save(output_path, results)
