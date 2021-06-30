import numpy as np
import argparse
from numba import njit, prange
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="Creates the constraint matrix for cube method")
parser.add_argument("chain_path", help="path of the MCMC chain in csv format")
parser.add_argument("gradient_path", help = "path of the MCMC gradient in csv format")
parser.add_argument("mcmcAlgo", help= "Type of MCMC algorithm used")
parser.add_argument("second_order", help="Should the second order constraints be added ?", type=int)

arguments = parser.parse_args()
h_x = np.genfromtxt(arguments.chain_path, delimiter=",")
h_grad = np.genfromtxt(arguments.gradient_path, delimiter=",")
output_path = arguments.mcmcAlgo
second_order = arguments.second_order


#@njit
#def create_constraint(i,j, chain, grad):
    #i, j are int
    # chain, grad are (N_particles, dimension)
#    if i == j:
#        return 2*chain[:,i] + chain[:, i]**2 * grad[:, i]

#    return chain[:, j] + chain[:, i]*chain[:, j] * grad[:, i]

@njit
def create_constraint(i,j, chain, grad):
    #i, j are int
    # chain, grad are (N_particles, dimension)
    if i == j:
        return 1 + chain[:, i]* grad[:, i]

    else:
        return chain[:, j]*grad[:, i]


@njit(parallel=True)
def create_constraint_matrix(chain, grad):
    # chain, grad are (N_particles, dimension)
    dimension = chain.shape[1]
    second_order_constraints = np.zeros((len(chain), dimension**2))
    for i in prange(dimension):
        for j in range(dimension):
            second_order_constraints[:, i*dimension+j] = create_constraint(i, j, chain, grad)

    return second_order_constraints



dimension = h_x.shape[1]
n_particles = len(h_x)
n_chains = 1


#x_times_grad = h_x * h_grad + 1
#all_As = np.concatenate([h_grad, x_times_grad], axis = 1)
all_As = h_grad
print("Constraints on mean done !")

if second_order:
    second_order_constraints = create_constraint_matrix(h_x, h_grad)
    constraints = np.concatenate([all_As, second_order_constraints], axis = 1)
else:
    constraints = all_As

print("Constraint dimensions:",constraints.shape)
d = {"constraints":constraints, "means":True, "variances":True}
np.save(output_path, d, allow_pickle=True)

