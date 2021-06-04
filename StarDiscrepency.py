import argparse
import time
import numpy as np
import utils



parser = argparse.ArgumentParser(description="Run star discrepency")
parser.add_argument("chain_path", help="path of the chain")
parser.add_argument("gradient_path", help="path of the gradient")
parser.add_argument("indexes_path", help="path of the indexes and weights")
parser.add_argument("output_path", help="path of the output weigths")
parser.add_argument("burnin", help="burnin of the full chain in order to have a good representation of the target distribution", type = int)


arguments = parser.parse_args()
chain_path = arguments.chain_path
grad_path = arguments.gradient_path
indexes = arguments.indexes_path
output_path = arguments.output_path
burnin = arguments.burnin

chain = np.genfromtxt(arguments.chain_path, delimiter=",")
gradient = np.genfromtxt(arguments.gradient_path, delimiter=",")
indexes = np.load(indexes, allow_pickle=True)
indexes = indexes.item()

sigma_chain = np.cov(chain[burnin:].T)
mean_chain = np.mean(chain[burnin:], axis=0)
sigma_chain_chol = np.linalg.cholesky(sigma_chain)

if "SMPCOV" in indexes.keys() or "SCLMED" in indexes.keys() or "MED" in indexes.keys():
    results = {}

    for method in indexes.keys():
        print(method)
        start = time.time()
        star_discrepency = utils.discrepency(chain[indexes[method]["indexes"]], chain[burnin:], sigma_chain_chol, mean_chain)
        results[method] = star_discrepency
        print("Star discrepency " + method + ": ", star_discrepency)
        print(time.time() - start)

    np.save(output_path, results, allow_pickle=True)


else:
    all_star = []
    weights = indexes["weights"]
    N_KEEP = indexes["M"]
    all_selected = indexes["all_selected"]
    burnin_cube = indexes["burnin"]
    n_experiments = len(all_selected)
    omega = np.sum(np.abs(weights))
    for i in range(n_experiments):
        print(i)
        start = time.time()
        selected = all_selected[i, :]
        chain_cube = chain[burnin_cube:][np.abs(selected) == 1]
        weights = selected[np.abs(selected) == 1]*omega/N_KEEP
        star_discrepency = utils.discrepency(chain_cube, chain[burnin:], sigma_chain_chol, mean_chain, weights=weights)
        all_star.append(star_discrepency)
        print("Star Discrepency " + ": ", star_discrepency)
        print(time.time() - start)


    np.save(output_path, {"star_discrepency":np.array(all_star)}, allow_pickle=True)


