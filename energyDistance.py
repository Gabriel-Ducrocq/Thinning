import time
import numpy as np
import argparse
from utils import *

parser = argparse.ArgumentParser(description="Runs the cube method for a specified number of times and with specific burnin")
parser.add_argument("chain_path", help="path of the chain")
parser.add_argument("indexes_path", help="path of the cube data")
parser.add_argument("output_path", help="path of the output weigths")
parser.add_argument("n_experiments", help="number of times cube method should be run", type=int)
parser.add_argument("burnin", help="number of initial particles to remove in order to avoid burnin", type=int)

arguments = parser.parse_args()
chain_path = arguments.chain_path
indexes_path = arguments.indexes_path
output_path = arguments.output_path
n_experiments = arguments.n_experiments
burnin = arguments.burnin

chain = np.genfromtxt(chain_path, delimiter =",")
indexes = np.load(indexes_path, allow_pickle=True)
indexes = indexes.item()

def energyDistance(chain, selected, signs, omega, N_KEEP,burnin_cube, burnin = 0, full_chain_dist = 0):
    sum_signed_measure = (omega/N_KEEP)*np.sum(signs[np.abs(selected) == 1])
    x = chain[burnin_cube:][np.abs(selected) == 1, :]
    y = chain[burnin:]
    dist_mat = compute_distance_matrix(x,y)
    auto_dist_x = compute_distance_matrix(x, x)
    signs_matrix = np.matmul(signs[np.abs(selected) == 1][:, None], signs[np.abs(selected) == 1][None, :])
    return 2*(omega/(len(chain)*N_KEEP*sum_signed_measure))* np.sum(signs[np.abs(selected) == 1]*np.sum(dist_mat, axis = 1)) - (omega/(N_KEEP*sum_signed_measure))**2*np.sum(
        signs_matrix*auto_dist_x) - full_chain_dist



all_star = []
weights = indexes["weights"]
N_KEEP = indexes["M"]
all_selected = indexes["all_selected"]
burnin_cube = indexes["burnin"]
all_signs = indexes["all_signs"]
n_experiments = len(all_selected)
omega = np.sum(np.abs(weights))
all_ED = []
for i in range(n_experiments):
    print(i)
    start = time.time()
    selected = all_selected[i, :]
    signs = all_signs[i, :]
    weights = selected[np.abs(selected) == 1]*omega/N_KEEP
    print(selected)
    print(np.sum(np.abs(selected)))
    print(chain.shape)
    print("Computing the energy distance...")
    ED = energyDistance(chain, selected, signs, omega, N_KEEP, burnin_cube, burnin)
    all_ED.append(ED)
    print("ED:", ED)
    print(time.time() - start)


np.save(output_path, {"energy_distance":np.array(all_ED)}, allow_pickle=True)
