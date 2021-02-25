import utils
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
from rpy2.robjects.packages import importr
from rpy2.robjects import r
from rpy2.robjects import FloatVector
import time
import argparse
import utils
importr('emplik')




d = np.load("data/Lotka/PRECOND-MALA/M100/constraintsCube.npy", allow_pickle = True)
d = d.item()
constraints = d["constraints"]

c = np.zeros(len(constraints))
A = np.r_[constraints.T, np.ones((1, len(constraints)))]
b = np.r_[np.zeros(constraints.shape[1]), np.ones(1)]
lp = linprog(c, A_eq=A, b_eq=b)

w = lp.x

print(np.sum(w))
print(np.dot(constraints.T, w))


print(lp.success)


constraintsR= r['matrix'](FloatVector(np.array(list(constraints.flatten()))), nrow=constraints.shape[0], ncol=constraints.shape[1], byrow=True)
means = r['numeric'](constraints.shape[1])
w = r['matrix'](FloatVector(np.ones(constraints.shape[0])), nrow=constraints.shape[0])

start = time.time()
result = r['el.test.wt2'](x=constraintsR, wt= w, mu=means, itertrace=False)
print("Duration:", time.time() - start)
weights = np.array(result[0])
n_iter = result[-1]
plt.plot(weights)
plt.show()

print(np.sum(weights))
print(np.dot(constraints.T, weights))
print("N_iter:", n_iter)

"""



t = 2000000

c = np.zeros(len(constraints[:t]))
A = np.r_[constraints[:t].T, np.ones((1, len(constraints[:t])))]
b = np.r_[np.zeros(constraints.shape[1]), np.ones(1)]
lp = linprog(c, A_eq=A, b_eq=b)

w = lp.x

print(np.sum(w))
print(np.dot(constraints[:t].T, w))


print(lp.success)
#test = ConvexHull(constraints[:t])

lambd = np.zeros(constraints.shape[1])
lambda_result = utils.newton_raphson(lambd, constraints[:t])


weights = (1/(np.dot(constraints[:t], lambda_result) + 1))*(1/len(constraints[:t]))



print(np.sum(weights))
print(np.dot(constraints[:t].T, weights))
plt.plot(weights)
plt.show()

"""