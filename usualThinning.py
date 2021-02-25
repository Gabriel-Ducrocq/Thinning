import numpy as np
import argparse
import utils
from stein_thinning.stein import kmat
from stein_thinning.kernel import make_imq


parser = argparse.ArgumentParser(description="Creates the constraint matrix for cube method")
parser.add_argument("chain_path", help="path of the MCMC chain in csv format")
parser.add_argument("gradient_path", help="path of the gradient")
parser.add_argument("output_path", help="path of the output weigths")
parser.add_argument("N_KEEP", help="Number of points to keep", type=int)
parser.add_argument("burnin", help="burnin of the full chain", type=int)



arguments = parser.parse_args()
chain = np.genfromtxt(arguments.chain_path, delimiter=",")
gradient = np.genfromtxt(arguments.gradient_path, delimiter=",")
M = arguments.N_KEEP
burnin = arguments.burnin
output_path = arguments.output_path

sigma_chain = np.cov(chain[burnin:].T)
mean_chain = np.mean(chain[burnin:], axis=0)


indexes = np.linspace(burnin, len(chain)-1, M, dtype=int)

star_discrepency = utils.discrepency(chain[indexes], chain[burnin:], sigma_chain, mean_chain)
vfk0 = make_imq(chain, gradient, pre='med')

print("Star discrepency done")
thin_mat = kmat(chain[indexes], gradient[indexes], vfk0)
KSD_thin = np.sqrt(np.mean(thin_mat))
print("KSD done")
ED_thin = 2*np.mean(utils.compute_distance_matrix(chain[indexes], chain[burnin:])) \
- np.mean(utils.compute_distance_matrix(chain[indexes], chain[indexes]))
print("ED done")


d = {"thinning":indexes, "ED":ED_thin, "KSD":KSD_thin, "star_discrepency":star_discrepency, "burnin":burnin}
np.save(output_path, d, allow_pickle=True)