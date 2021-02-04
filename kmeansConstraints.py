import numpy as np
from rpy2.robjects.packages import importr
from rpy2.robjects import r
from rpy2.robjects import FloatVector
import matplotlib.pyplot as plt
import time
from utils import project_one_chain, flight_phase
import config



def dist_between_centroids(centroids, chain):
    dist = np.zeros((len(centroids), len(centroids)))
    dist_centroids_points = np.zeros((len(centroids), len(chain)))
    for i in range(len(centroids)):
        dist[i, :] = np.sqrt(np.sum((centroids[i, :] - centroids)**2, axis=1))
        dist[i, i] = np.inf

        dist_centroids_points[i, :] = np.sqrt(np.sum((centroids[i, :] - chain)**2, axis = 1))

    return dist, dist_centroids_points

def centroids_radiuses(dist):
    return np.min(dist, axis = 1)/2

def compute_centroids_distances(centroids, points, clusters):
    ## centroids is (N_centroid, dimension)
    ## points is (N_points, dimension)
    ## return: (N_cluster, N_points) and (n_clusters,)
    distances = np.zeros((len(centroids), len(points)))
    radiuses = np.zeros(len(centroids))
    for i in range(len(centroids)):
        dist = np.sqrt(np.sum((centroids[i, :] - points)**2, axis = 1))
        distances[i, :] = dist
        radiuses[i] = np.max(dist[np.array(clusters) == i+1])

    return distances, radiuses

def regulate_radius(distances, radiuses):
    d = distances[:]
    new_radiuses = radiuses[:]
    d[np.where(distances > radiuses.reshape((len(radiuses), 1)))] = np.inf
    cross_points = np.sum((d < np.inf), axis=0)
    for i in range(distances.shape[1]):
        if cross_points[i] > 1:
            new_radiuses = np.minimum(new_radiuses, distances[:, i])

    return new_radiuses

def get_number_activated_constraints(distances, radiuses):
    ## distances is (n_clusters, n_points)
    ## radiuses is (n_clusters, 1)
    d = np.maximum(radiuses.reshape((len(radiuses), 1)) - distances, 0)
    activated = d > 0
    appartenance_clust = np.sum(activated, axis=0)
    print("Results")
    print(appartenance_clust)
    print(np.unique(appartenance_clust))
    print(np.sum(appartenance_clust))
    print("Percentage of point:", np.sum(appartenance_clust)/distances.shape[1])


def get_radiuses_clusters(points, gradients, centroids, clusters):
    gradients = gradients.T
    constraints = []
    all_radiuses = np.zeros(len(clusters))
    for i in range(len(centroids)):
        differences = (points[clusters == i + 1, :] - centroids[i, :]).T
        dist = np.sqrt(np.sum((differences)**2, axis = 0))
        radius = np.max(dist)
        all_radiuses[i] = radius
        constraint = (-4*differences + gradients[:, clusters == i + 1])*(radius - dist**2)**2
        constraints.append(constraint)

    return constraints, all_radiuses




d = np.load("chain.npy", allow_pickle=True)
d = d.item()
h_x = d["h_x"]
h_grad = d["h_grad"]
burnin = 1000

n_clusters = 10000
all_clusters = []
chain = h_x[burnin:,:]
h_grad = h_grad[burnin:, :]
path_R = r['matrix'](FloatVector(np.array(list(h_x[burnin:, :].T.flatten()))), nrow=h_x[burnin:, :].shape[0])
all_signs = []
all_selected = []
all_weights = []



all_constraints = np.zeros((50, n_clusters, len(chain)))
epsilon = 1e-10
start = time.time()
for m in range(50):
    print(m)
    resKMeans = r["kmeans"](path_R, n_clusters, 30)
    cluster = dict(zip(resKMeans.names, list(resKMeans)))["cluster"]
    cluster = np.array([r for r in cluster])
    centroids = dict(zip(resKMeans.names, list(resKMeans)))["centers"]
    centroids = np.array([r for r in centroids])
    centroids = centroids.reshape(-1, n_clusters, order='C').T
    all_clusters.append(cluster)
    full_constraints = np.zeros((n_clusters, len(chain)))
    all_distances = np.zeros((n_clusters, len(chain)))

    #blocks, radiuses = get_radiuses_clusters(chain, h_grad, centroids, cluster)

    #dist_between_centroids = np.zeros((n_clusters, n_clusters))

    #distances, radiuses = compute_centroids_distances(centroids, chain, cluster)
    #new_radiuses = regulate_radius(distances, radiuses)
    dist_centroids, dist_points = dist_between_centroids(centroids, chain)
    radiuses = centroids_radiuses(dist_centroids)
    get_number_activated_constraints(dist_points, radiuses)
    for i in range(1, n_clusters+1):
        """
        dist_square = np.sum((centroids[i - 1, :] - chain) ** 2, axis=1)
        radius = np.max(dist_square[cluster == i])
        all_distances[i-1, :] = np.maximum(dist_square, 0)
        distances = all_distances[:]
        all_distances[i - 1, dist_square >= radius] = np.inf
        dist_between_centroids[i-1, :] = np.sqrt(np.sum((centroids[i - 1, :] - centroids) ** 2, axis=1))
        dist_between_centroids[i-1, i-1] = np.inf
        """
        #deriv = np.sum(chain - centroids[i - 1, :], axis=1)
        #first_term = -4*np.maximum(radius - dist_square, 0) * deriv
        #factor = (np.maximum(0, radius - dist_square)**2).reshape(len(chain), 1)
        #second_term = np.sum(factor * h_grad[burnin:, :], axis=1)
        #constraints = first_term + second_term
        #full_constraints[i-1, :] = constraints


    A = full_constraints
    projected_point = project_one_chain(A, np.ones(A.shape[1]) * 0.5)
    projected_point /= np.sum(projected_point)
    all_weights.append(projected_point)

    start = time.time()
    print("Starting Cube:")
    A = np.vstack([np.ones(A.shape[1]), A])
    selected, signs = flight_phase(A.T, projected_point)
    all_signs.append(signs)
    all_selected.append(selected*signs)
    end = time.time()
    print("Duration:", end - start)
    print(np.sum(np.abs(projected_point)) * np.sum(h_x[burnin:, :].T * selected * signs / config.N_KEEP, axis=1))


all_signs = np.array(all_signs)
all_selected = np.array(all_selected)
all_clusters = np.array(all_clusters)
all_weights = np.array(all_weights)

end = time.time()

d = {"clusters":all_clusters, "all_signs":all_signs, "all_selected":all_selected, "burnin":burnin, "weights":all_weights}
np.save("kmeansResults.npy", d, allow_pickle=True)




