import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

d = np.load("official_run.npy", allow_pickle=True)
d = d.item()

sigma_target = d["sigma_target"]
mu_target = d["mu_target"]
burnin = 10000
h_x = d["h_x"]
ind =2
xx = np.linspace(mu_target[ind] - 10, mu_target[ind] + 10, 10000)
yy = np.array([norm.pdf(x, loc = mu_target[ind], scale = np.sqrt(sigma_target[ind, ind])) for x in xx])

print(sigma_target[ind, ind])
print("\n")
print(sigma_target)
plt.hist(h_x[10000:, ind], density=True, bins = 50)
plt.plot(xx, yy)
plt.show()

plt.plot(h_x[:, ind])
plt.show()
