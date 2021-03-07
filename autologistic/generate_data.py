import numpy as np
import utils_autologistic
import argparse


parser = argparse.ArgumentParser(description="generate autologistic data")
parser.add_argument("alpha", help="alpha", type=float)
parser.add_argument("beta", help="beta", type=float)
parser.add_argument("vertex_number", help="number of points on the map", type=int)
parser.add_argument("N_gibbs_step", help="number of gibbs step", type=int)
parser.add_argument("output_path", help="path of the output weigths")


arguments = parser.parse_args()

alpha = arguments.alpha
beta = arguments.beta
vertex_number = arguments.vertex_number
n_gibbs = arguments.N_gibbs_step
output_path = arguments.output_path


neighbour_matrix = np.zeros((vertex_number, vertex_number))
neighbour_matrix = utils_autologistic.fill(neighbour_matrix)

y = np.random.binomial(1, 0.5, vertex_number) * 2 - 1
y = utils_autologistic.run_gibbs_wrapper(y.copy(), alpha, beta, neighbour_matrix, n_iter=n_gibbs, history=False)


d = {"data":y, "neighbours":neighbour_matrix, "alpha":alpha, "beta":beta, "n_gibbs":n_gibbs}

np.save(output_path, d, allow_pickle=True)