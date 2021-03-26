from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np
import utils_truncated
import sys
from scipy.stats import truncnorm, norm
import argparse
import statsmodels.api as sm
import os, sys
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
mu = np.zeros(len(mu_true))+ 5
#mu = 5
sigma = np.eye(len(mu_true))*1
#sigma =2

intercepts = []
all_means = []
#h_true = truncnorm.rvs(-mu_true/np.sqrt(sigma_true), np.inf, mu_true, np.sqrt(sigma_true), size = 100000)
#regressors_true = np.vstack([h_true, -(1/2)*h_true**2]).T
for k in range(n_experiments):
    h_x = []
    x = np.ones(len(mu))*2
    print(k)
    #for i in range(n_iter):
    #    l = np.random.randint(0, len(mu))
    #    x[l] = utils_truncated.gibbs_step(l, x.copy(), mu, sigma)
    #    h_x.append(x.copy())
    #h_x = truncnorm.rvs(-mu/np.sqrt(sigma), np.inf, mu, np.sqrt(sigma), size = 100000)
    #h_true = np.random.normal(mu_true, np.sqrt(sigma_true), 100000)
    #h_x = np.random.normal(mu, np.sqrt(sigma), size = 100000)
    h_y = np.zeros((n_iter, 3))
    h_y[:, 0] = truncnorm.rvs(-mu_true[0]/np.sqrt(sigma_true[0, 0]), np.inf, mu_true[0], np.sqrt(sigma_true[0, 0]), size=len(h_y))
    h_y[:, 1] = truncnorm.rvs(-mu_true[1]/np.sqrt(sigma_true[1, 1]), np.inf, mu_true[1], np.sqrt(sigma_true[1, 1]), size=len(h_y))
    h_y[:, 2] = truncnorm.rvs(-mu_true[2]/np.sqrt(sigma_true[2,2]), np.inf, mu_true[2], np.sqrt(sigma_true[2, 2]), size=len(h_y))
    h_x = np.zeros((n_iter, 3))
    h_x[:, 0] = truncnorm.rvs(-mu[0]/np.sqrt(sigma[0, 0]), np.inf, mu[0], np.sqrt(sigma[0, 0]), size=len(h_x))
    h_x[:, 1] = truncnorm.rvs(-mu[1]/np.sqrt(sigma[1, 1]), np.inf, mu[1], np.sqrt(sigma[1, 1]), size=len(h_x))
    h_x[:, 2] = truncnorm.rvs(-mu[2]/np.sqrt(sigma[2,2]), np.inf, mu[2], np.sqrt(sigma[2, 2]), size=len(h_x))
    #h_x = np.array(h_x)
    print(mu_true)
    print(mu)
    print("Sigma:")
    print(sigma_true)
    print(sigma)
    #regressors_false = np.vstack([h_x, -(1/2)*h_x**2]).T
    #regressors_true = np.vstack([h_y, -(1/2)*h_y**2]).T
    #control_variates = utils_truncated.construct_cv(h_x.copy(), mu, sigma)
    #weights = get_weights(control_variates)
    #control_variates = np.hstack((np.ones((len(control_variates), 1)),control_variates))
    #print(weights.shape)
    #print(np.dot(control_variates.T, weights))
    #print(regressors_true.shape)
    #result = cube_method(control_variates, weights, M)
    triu_indices = np.triu_indices(h_x.shape[1])
    regressors_false = utils_truncated.compute_all_regressors(h_x, triu_indices)
    regressors_true = utils_truncated.compute_all_regressors(h_y, triu_indices)
    labels = np.zeros(len(regressors_false)+len(regressors_true))
    labels[:len(regressors_true)] = 1
    #print(regressor_true.shape)
    #print(regressor_false.shape)
    all_points =  np.vstack((regressors_true, regressors_false))
    print(all_points.shape)
    print(labels.shape)
    logit = LogisticRegression(solver="newton-cg",penalty="none", fit_intercept = True)
    logit.fit(all_points, labels)
    print(logit.coef_)
    intercepts.append(logit.coef_)
    #print(np.linalg.solve(sigma_true, mu_true))
    #X = sm.add_constant(control_variates)
    #reg = sm.OLS(h_x[:, 0], X)
    #results = reg.fit()
    #intercept = results.params[0]
    #intercepts.append(intercept)
    #print(intercept)
    #print(np.mean(h_x[:, 0]))
    #all_means.append(np.mean(h_x[:, 0]))

intercepts = np.array(intercepts)
print("Intercept")
print("Mean", np.mean(intercepts, axis = 0))
print("Std", np.std(intercepts, axis = 0))
print("Minimum", np.min(intercepts, axis = 0))
#print("Usual")
#print("Mean", np.mean(all_means))
#print("Var", np.var(all_means))
#a = truncnorm.rvs(-mu[0]/np.sqrt(sigma[0, 0]), np.inf, mu[0], np.sqrt(sigma[0, 0]), size = 10000)
#print(np.mean(a))
