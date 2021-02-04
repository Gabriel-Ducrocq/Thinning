import numpy as np
import config




d = np.load("cube_nicolas.npy", allow_pickle=True)

d = d.item()
all_signs = d["all_signs"]
all_selected = d["all_selected"]
weights = d["weights"]
burnin = d["burnin"]

d = np.load("chain.npy", allow_pickle=True)
d = d.item()
h_x = d["h_x"]


dim_interest = 2
chain = h_x[burnin:, dim_interest]
omega = np.sum(np.abs(weights))
all_estim = np.sum(chain*all_selected, axis = 1)*omega/config.N_KEEP
print(np.mean(all_estim))
print(np.std(all_estim))

print(np.mean(chain**2))
all_estim_var = np.sum(chain**2*all_selected, axis = 1)*omega/config.N_KEEP
print(np.mean(all_estim_var))
print(np.std(all_estim_var))





