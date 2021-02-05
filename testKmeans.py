import utils
import numpy as np
import matplotlib.pyplot as plt


d = np.load("data/Lotka/RW/constraintsMeansOnly.npy", allow_pickle = True)
d = d.item()
constraints = d["constraints"]

lambd = np.zeros(constraints.shape[1])
lambda_result = utils.newton_raphson(lambd, constraints)


weights = (1/(np.dot(constraints, lambda_result) + 1))*(1/len(constraints))



print(np.sum(weights))
print(np.dot(constraints.T, weights))
plt.plot(weights)
plt.show()