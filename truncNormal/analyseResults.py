import numpy as np


d =np.load("estimFull.npy", allow_pickle=True)
d = d.item()
sigma_true = d["sigma_true"]
mu_true = d["mu_true"]
sigma_false = d["sigma_false"]
mu_false = d["mu_false"]
coefficients_full = d["estim_coeffs"]
intercept_full = d["intercepts"]


d =np.load("estimCube.npy", allow_pickle=True)
d = d.item()
coefficients_cube = d["estim_coeffs"]
intercept_cube = d["intercepts"]

d =np.load("estimThin.npy", allow_pickle=True)
d = d.item()
coefficients_thin = d["estim_coeffs"]
intercept_thin = d["intercepts"]


Q_true = np.linalg.inv(sigma_true)
r_true = np.linalg.solve(sigma_true, mu_true)
Q_false = np.linalg.inv(sigma_false)
r_false = np.linalg.solve(sigma_false, mu_false)

full_estims = np.mean(coefficients_full, axis = 0)
full_std = np.std(coefficients_full, axis = 0)

cube_estims = np.mean(coefficients_cube, axis = 0)
cube_std = np.std(coefficients_cube, axis = 0)

thin_estims = np.mean(coefficients_thin, axis = 0)
thin_std = np.std(coefficients_thin, axis = 0)

cube_estims = np.mean(coefficients_cube, axis = 0)
cube_std = np.std(coefficients_cube, axis = 0)

r_full_estim = full_estims[:len(r_true)]
r_full_std = full_std[:len(r_true)]

triu_indices = np.triu_indices(len(r_full_std))
Q_full_estim = full_estims[len(r_true):]
Q_full_std = full_std[len(r_true):]

Q_thin_estim = thin_estims[len(r_true):]
Q_thin_std = thin_std[len(r_true):]

r_cube_estim = cube_estims[:len(r_true)]
r_cube_std = cube_std[:len(r_true)]
Q_cube_estim = cube_estims[len(r_true):]
Q_cube_std = cube_std[len(r_true):]

r_thin_estim = thin_estims[:len(r_true)]
r_thin_std = thin_std[:len(r_true)]
Q_thin_estim = thin_estims[len(r_true):]
Q_thin_std = thin_std[len(r_true):]

print("Full NCE:")
print("--> Estimation of r")
print(r_true)
print(r_full_estim + r_false)
print(r_full_std)
print("--> Estimation of Q")
print(Q_true[triu_indices])
print(Q_full_estim + Q_false[triu_indices])
print(Q_full_std)

print("\n")
print("Cube Method:")
print(r_true)
print(r_cube_estim + r_false)
print(r_cube_std)
print("--> Estimation of Q")
print(Q_true[triu_indices])
print(Q_cube_estim + Q_false[triu_indices])
print(Q_cube_std)

print("\n")
print("Thin Method:")
print(r_true)
print(r_thin_estim + r_false)
print(r_thin_std)
print("--> Estimation of Q")
print(Q_true[triu_indices])
print(Q_thin_estim + Q_false[triu_indices])
print(Q_thin_std)

