import numpy as np
from rpy2.robjects.packages import importr
from rpy2.robjects import r
from rpy2.robjects import FloatVector
import time
import matplotlib.pyplot as plt
import argparse
import utils
importr('BalancedSampling')
importr('emplik')



parser = argparse.ArgumentParser(description="Runs the cube method for a specified number of times and with specific burnin")
parser.add_argument("chain_path", help="path of the chain")
parser.add_argument("constraints_path", help="path of the constraints")
parser.add_argument("output_path", help="path of the output weigths")
parser.add_argument("N_KEEP", help="number of particles to keep after compression", type=int)
parser.add_argument("n_experiments", help="number of times cube method should be run", type=int)
parser.add_argument("burnin", help="number of initial particles to remove in order to avoid burnin", type=int)
parser.add_argument("emplik", help="Should we use the empirical likelihood way of finding the weights", type=int)
parser.add_argument("burnin_ed", help="Burnin for the approximation of the energy distance", type= int)

arguments = parser.parse_args()
chain_path = arguments.chain_path
path = arguments.constraints_path
output_path = arguments.output_path
N_KEEP = arguments.N_KEEP
n_experiments = arguments.n_experiments
burnin = arguments.burnin
emplik = arguments.emplik
burnin_ed = arguments.burnin_ed

##Regression linéaire avec fonction quelcondque f(X) = Y.
##essayer avec la ligne d'intercept, regarder Art Owens.
## Regarder le R2

def project_one_chain(A, point):
    ### A is (N_constraints, N_particles) and point is (N_particles)
    return point - np.matmul(A.T,np.linalg.solve(np.matmul(A, A.T), np.matmul(A, point)))


def compute_weights(A):
    ### A is (N_constraints, N_particles)
    S_hh = np.matmul(A, A.T)
    H = np.sum(A, axis = 1)
    return (1 - np.matmul(A.T,np.linalg.solve(S_hh, H)))/A.shape[1]


def test_weights(A):
    A = A.T
    weights = np.zeros(A.shape[1])
    H = np.mean(A, axis=1)
    S_hh = np.matmul(A-H[:, None], (A-H[:, None]).T)
    d = np.linalg.solve(S_hh, H)
    for i in range(len(weights)):
        weights[i] = np.dot(A[:,i]-H, d)

    return 1/len(weights) - weights


def test2(A):
    H = np.sum(A, axis= 1)
    Q = np.linalg.solve(np.matmul(A - H[:, None], (A-H[:, None]).T), A - H[:, None])
    return Q


def get_constrained_weights(constraints):
    ## A is (N_particles, N_constraints)
    constraintsR = r['matrix'](FloatVector(np.array(list(constraints.flatten()))), nrow=constraints.shape[0],
                               ncol=constraints.shape[1], byrow=True)
    means = r['numeric'](constraints.shape[1])
    w = r['matrix'](FloatVector(np.ones(constraints.shape[0])), nrow=constraints.shape[0])
    result = r['el.test.wt2'](x=constraintsR, wt=w, mu=means, itertrace=False)
    weights = np.array(result[0])
    return weights


def cube_method(A, point, N_KEEP):
    ### A is (N_constraints, N_particles) and point is (N_particles)
    signs = np.sign(point)
    A_signs = A.T.copy()
    A_signs[1:, :] = A_signs[1:, :]*signs
    point_signs = np.abs(point)*N_KEEP/np.sum(np.abs(point))
    A_signs *= point_signs
    A_signs_R = r['matrix'](FloatVector(np.array(list(A_signs.flatten()))), nrow=A_signs.T.shape[0])
    print("Flight phase:")
    res = r["flightphase"](FloatVector(point_signs), A_signs_R)
    print("Landing Phase:")
    land_point = r["landingphase"](FloatVector(point_signs), res, A_signs_R)
    return np.array([r for r in land_point]), signs


chain = np.genfromtxt(arguments.chain_path, delimiter=",")

d = np.load(path, allow_pickle=True)
d = d.item()
A = d["constraints"]

A = A[burnin:, :]
#chain = chain[burnin:, :]
all_selected = []
all_signs = []
all_weights = []
all_projected = []


#A = np.hstack([np.ones((len(A), 1)), A])

#projected_point_bis = project_one_chain(A.T, np.ones(A.shape[0])/len(A))
#projected_point_bis /= np.sum(projected_point_bis)
if emplik == 0:
    projected_point = test_weights(A)
else:
    projected_point = get_constrained_weights(A)

#Q = test2(A.T)
#dim = 6
#beta_hat = np.dot(Q, A[:, dim])
#predict = np.dot(A, beta_hat)
#SS_est = np.mean((A[:, dim] - predict)**2)
#SS_tot = np.var(A[:, dim])
#print("R2:", 1- SS_est/SS_tot)
#print(sum(proj_weights_ter))
#print(np.dot(A.T, proj_weights_ter))
print("SUM OF PROJECTED POINTS:", np.sum(projected_point))
print("Constraints respected ?", np.dot(A.T, projected_point))
#print("Are they equivalent ?:", projected_point, projected_point_bis)
plt.plot(projected_point)
plt.show()

#projected_point /= np.sum(projected_point)
#print("Percentage of negative weights:",np.mean(projected_point <= 0))
#print("SUM OF PROJECTED POINTS:", np.sum(projected_point))

A = np.hstack([np.ones((A.shape[0], 1)), A])
all_ED = []
duration_cube = []
for k in range(n_experiments):
    print(k)
    start = time.time()
    cpu_start = time.clock()
    print("Starting Cube:")
    selected, signs = cube_method(A, projected_point, N_KEEP)
    duration_cube.append(time.clock() - cpu_start )
    print("Duration Cube:", time.time() - start)
    all_signs.append(signs)
    all_selected.append(selected*signs)
    omega = np.sum(np.abs(projected_point))
    ED = utils.energyDistance(chain, selected, signs,omega, N_KEEP, burnin, burnin_ed)
    all_ED.append(ED)
    end = time.time()
    print("ED:", ED)
    print("Duration:", end - start)
    print("\n")


all_signs = np.array(all_signs)
all_selected = np.array(all_selected)
all_ED = np.array(all_ED)
duration_cube = np.array(duration_cube)

d = {"all_signs": all_signs, "all_selected":all_selected, "weights":projected_point, "burnin":burnin, "M":N_KEEP,
     "ED":all_ED, "duration_cube":duration_cube, "emplik":emplik}
np.save(output_path, d, allow_pickle=True)
