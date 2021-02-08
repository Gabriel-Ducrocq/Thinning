import argparse
import utils
from stein_thinning.thinning import thin
import time
import numpy as np




parser = argparse.ArgumentParser(description="Run stein thinning")
parser.add_argument("chain_path", help="path of the chain")
parser.add_argument("grad_path", help="path of the gradient")
parser.add_argument("output_path", help="path of the output weigths")
parser.add_argument("N_KEEP", help="number of particles to keep after compression", type=int)


arguments = parser.parse_args()
chain_path = arguments.chain_path
grad_path = arguments.grad_path
output_path = arguments.output_path
N_KEEP = arguments.N_KEEP

chain = np.genfromtxt(arguments.chain_path, delimiter=",")
grad = np.genfromtxt(arguments.grad_path, delimiter=",")

def run_KSD_thinning(chain, grad, kernel="smpcov", N_KEEP = 100, full_chain_dist = 0, sigma = None, burnin = 0):
    start = time.time()
    indexes = thin(chain, grad, N_KEEP, pre=kernel)
    print("Time to Thinning " + kernel, time.time() - start)
    chain_noBurnin = chain[burnin:, :]
    ST_chain = chain[indexes, :]
    ST_distance_mat = utils.compute_distance_matrix(ST_chain, chain_noBurnin, sigma)
    ST_auto_distance = utils.compute_distance_matrix(ST_chain, ST_chain, sigma)
    ST_ED = 2*np.sum(ST_distance_mat)/(len(chain_noBurnin)*N_KEEP) - np.sum(ST_auto_distance)/N_KEEP**2 - full_chain_dist
    print("Time to ED " + kernel, time.time() - start)
    return ST_ED, indexes





ST_ED_smpcov, indexes_smpcov = run_KSD_thinning(chain, grad, kernel="smpcov", N_KEEP = N_KEEP)
ST_ED_sclmed, indexes_sclmed = run_KSD_thinning(chain, grad, kernel="sclmed", N_KEEP = N_KEEP)
ST_ED_med, indexes_med = run_KSD_thinning(chain, grad, kernel="med", N_KEEP=N_KEEP)



d = {"SMPCOV":{"ED": ST_ED_smpcov, "indexes": indexes_smpcov}, "SCLMED":{"ED": ST_ED_sclmed, "indexes": indexes_sclmed},
     "MED":{"ED": ST_ED_med, "indexes": indexes_med}}


np.save(output_path, d, allow_pickle=True)