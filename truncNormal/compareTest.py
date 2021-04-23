import numpy as np

d = np.load("testThin.npy", allow_pickle=True)
d = d.item()
thin_mean = d["estim_means"]
thin_cov = d["all_covariances"]

d = np.load("testCube.npy", allow_pickle=True)
d = d.item()
cube_mean = d["estim_means"]
cube_cov = d["all_covariances"]

print("Means Thin:")
print(np.mean(thin_mean, axis = 0))
print(np.std(thin_mean, axis=0))


print("Means Cube:")
print(np.mean(cube_mean, axis = 0))
print(np.std(cube_mean, axis=0))

print("Cov Thin:")
print(np.mean(thin_cov, axis = 0))
print(np.std(thin_cov, axis = 0))

print("Cov Cube:")
print(np.mean(cube_cov, axis = 0))
print(np.std(cube_cov, axis = 0))
