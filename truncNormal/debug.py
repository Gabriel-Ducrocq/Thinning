import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from scipy.stats import truncnorm



mu_true = 10
mu_false =5
sigma_true = 1
sigma_false =1
all_fitted = []
x_true = truncnorm.rvs(-mu_true/np.sqrt(sigma_true), np.inf, mu_true, np.sqrt(sigma_true), size = 100000)
covariates_true = np.vstack([x_true, -(1/2)*x_true**2]).T
for l in range(100):
    x_false = truncnorm.rvs(-mu_false/np.sqrt(sigma_false), np.inf, mu_false, np.sqrt(sigma_false), size = 100000)

    covariates_false = np.vstack([x_false,-(1/2)*x_false**2]).T

    labels = np.zeros(200000)
    labels[:100000] = 1


    all_points =  np.vstack((covariates_true, covariates_false))
    logit = LogisticRegression(solver="newton-cg",penalty="none", fit_intercept = True)
    logit.fit(all_points, labels)
    print(logit.coef_)
    all_fitted.append(logit.coef_)

all_fitted = np.array(all_fitted)

print(np.mean(all_fitted, axis = 0))
print(np.std(all_fitted, axis = 0))
