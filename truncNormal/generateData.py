import matplotlib.pyplot as plt
from utils_truncated import gibbs_step, compute_all_regressors
import numpy as np
from scipy.stats import truncnorm, norm



mu = np.array([0, 0, 0])+10
#mu = 10
Sigma =np.eye(3)
#Sigma = 1
h_x = []

x =np.array([1.0, 1.0, 1.0])
all_points = []
for l in range(1000):
    print(l)
    h_x = []
    for _ in range(10000):
        i = np.random.randint(0, 3)
        x[i] = gibbs_step(i, x, mu, Sigma)
        h_x.append(x.copy())
    #x = truncnorm.rvs(-mu/np.sqrt(Sigma), np.inf, mu, np.sqrt(Sigma))
    #x = np.random.normal()*np.sqrt(Sigma) + mu
    h_x = np.array(h_x)
    #print(np.mean(h_x[:, 0]))
    print(x)
    all_points.append(x.copy())


all_points = np.array(all_points)
#all_points = np.random.normal(mu, np.sqrt(Sigma), size = 10000)
#triu_index = np.triu_indices(all_points.shape[1])
#regressors = compute_all_regressors(all_points,triu_index)
regressors = np.vstack([all_points, -(1/2)*all_points**2]).T
print(regressors)
d = {"points": all_points, "mu":mu, "Sigma":Sigma, "regressors":regressors}
np.save("dataset.npy", d, allow_pickle=True)
#print(all_points[:, 0])
#a = truncnorm.rvs(-mu[0]/np.sqrt(Sigma[0, 0]), np.inf, mu[0], np.sqrt(Sigma[0, 0]), size = 1000)
#plt.hist(all_points[:, 0], alpha = 0.5, density=True, bins = 15)
#plt.hist(a, alpha = 0.5, density=True, bins = 15)
#plt.show()
