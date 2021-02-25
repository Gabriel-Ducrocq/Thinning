import numpy as np
import utils_autologistic
from statsmodels.discrete.discrete_model import Logit
from sklearn.linear_model import LogisticRegression
import time
import matplotlib.pyplot as plt

n_points = 10
neighbour_matrix = np.zeros((n_points, n_points))
neighbour_matrix = utils_autologistic.fill(neighbour_matrix)



### Setting
alpha = 0.1
beta = 0.04

all_coefs = []
all_intercept = []
start = time.time()

for k in range(200):
    print("Iteration:", k)
    y = np.random.binomial(1, 0.5, n_points)*2 - 1
    all_y_true = []
    for i in range(1000):
        y = utils_autologistic.run_gibbs(y, alpha, beta, neighbour_matrix, n_iter=500, history=False)
        all_y_true.append(y)


    all_y_true = np.array(all_y_true)

    y0 = y
    print("\n")
    print(np.mean(y0 == 1))
    print(np.mean(y0 == -1))



    ### Run NCE

    alpha_nce = 0.01
    beta_nce = 0.05
    y_init = np.random.binomial(1, 0.5, n_points)*2 - 1

    all_y_false = []
    for _ in range(1000):
        y = utils_autologistic.run_gibbs(y_init, alpha_nce, beta_nce, neighbour_matrix, n_iter=500, history=False)
        all_y_false.append(y)


    all_y_false = np.array(all_y_false)

    print("False is done !")

    #constraints = utils_autologistic.create_constraints(all_y_false, alpha_nce, beta_nce, neighbour_matrix)

    #print(constraints.shape)

    feature_matrix_true = utils_autologistic.compute_dataset_regression(all_y_true, neighbour_matrix)
    feature_matrix_false = utils_autologistic.compute_dataset_regression(all_y_false, neighbour_matrix)


    dataset = np.vstack([feature_matrix_true, feature_matrix_false])
    label = np.zeros(len(dataset))
    label[:len(feature_matrix_false)] = 1


    regressor = LogisticRegression(penalty='none', solver = "lbfgs", fit_intercept=True)
    regressor.fit(dataset, label)
    coefs = regressor.coef_
    intercept = regressor.intercept_

    all_coefs.append(coefs)
    all_intercept.append(intercept)

print("Total time:", time.time() - start)

all_coefs = np.array(all_coefs)
all_intercept = np.array(all_intercept)
d = {"coefs":all_coefs, "intercepts":all_intercept, "alpha":alpha, "beta":beta, "alpha_nce": alpha_nce,
     "beta_nce":beta_nce, "neighbour":neighbour_matrix}

np.save("test.npy", d, allow_pickle=True)

alpha_nce = 0.01
beta_nce = 0.05
d = np.load("test.npy", allow_pickle=True)
d = d.item()

coefs = d["coefs"]

print(coefs.shape)


fig, ax = plt.subplots(1, 2)
ax[0].hist(coefs[:, 0, 0] + alpha_nce, alpha=0.5)
ax[0].axvline(x=alpha, color="red")

ax[1].hist(coefs[:, 0, 1] + beta_nce, alpha=0.5)
ax[1].axvline(x=beta, color="red")
plt.show()