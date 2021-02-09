from stein_thinning.stein import ksd, kmat
import argparse
import utils
from stein_thinning.thinning import thin
from stein_thinning.kernel import make_imq
import time
import numpy as np
import time


parser = argparse.ArgumentParser(description="Run stein thinning")
parser.add_argument("chain_path", help="path of the chain")
parser.add_argument("gradient_path", help="path of the gradient")
parser.add_argument("indexes_path", help="path of the indexes and weights")
parser.add_argument("output_path", help="path of the output weigths")


arguments = parser.parse_args()
chain = np.genfromtxt(arguments.chain_path, delimiter=",")
gradient = np.genfromtxt(arguments.gradient_path, delimiter=",")

indexes_path = arguments.indexes_path
output_path = arguments.output_path
indexes = np.load(indexes_path, allow_pickle=True)
indexes = indexes.item()
## Construct the stein kernel use for the computation of the KSD:
vfk0 = make_imq(chain, gradient, pre='med')


if "SMPCOV" in indexes.keys() or "SCLMED" in indexes.keys() or "MED" in indexes.keys():
    results = {}

    for method in indexes.keys():
        print(method)
        start = time.time()
        ## Computing k_stein(x,y)
        thin_mat = kmat(chain[indexes[method]["indexes"]], gradient[indexes[method]["indexes"]], vfk0)
        KSD_thin = np.sqrt(np.mean(thin_mat))
        results[method] = KSD_thin
        print("KSD " + method + ": ", KSD_thin)
        print(time.time() - start)

    np.save(output_path, results, allow_pickle=True)


else:
    all_KSD = []
    weights = indexes["weights"]
    N_KEEP = indexes["N_KEEP"]
    all_selected = indexes["all_selected"]
    n_experiments = len(all_selected)
    omega = np.sum(np.abs(weights))
    for i in range(all_selected):
        print(i)
        start = time.time()
        selected = all_selected[i, :]
        thin_mat = kmat(chain[np.abs(selected) == 1], gradient[np.abs(selected) == 1], vfk0)
        result = np.sqrt(np.sum(selected[:, None]*thin_mat*selected[None, :])*(omega/N_KEEP))
        all_KSD.append(result)
        print("KSD " +  ": ", result)
        print(time.time() - start)


    np.save(output_path, {"KSD":np.array(all_KSD)}, allow_pickle=True)
