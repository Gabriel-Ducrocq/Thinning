from sklearn.linear_model import LinearRegression, LogisticRegression
import matplotlib.pyplot as plt
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


arguments = parser.parse_args()
data = np.load(arguments.datapath, allow_pickle=True)
data = data.item()
mu_true = data["mu"]
sigma_true = data["Sigma"]
regressors_true = data["regressors"]
data = data["points"]

mu = mu_true[0]
print("Mu:", mu)
sigma = sigma_true[0, 0]

trunc = truncnorm(-mu/np.sqrt(sigma), np.inf, loc = mu, scale = np.sqrt(sigma))
x = np.linspace(0, 5, 10000)
y = np.array([trunc.pdf(xx) for xx in x])

plt.hist(data[:,0], density=True, alpha = 0.5, bins = 10)
plt.plot(x,y)
plt.show()
