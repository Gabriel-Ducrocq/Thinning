import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Convert npy to csv")
parser.add_argument("chain_path", help="path of the chain to use as preliminary")
arguments = parser.parse_args()

path = arguments.chain_path

d = np.load(path, allow_pickle=True)
d = d.item()
h_x = d["h_x"]
grad_x = d["h_grad"]

np.savetxt("theta.csv", h_x, delimiter =",", encoding="utf-8")
np.savetxt("grad.csv", grad_x, delimiter=",", encoding="utf-8")
