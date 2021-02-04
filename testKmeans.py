import numpy as np
from rpy2.robjects.packages import importr
from rpy2.robjects import r
from rpy2.robjects import FloatVector
import matplotlib.pyplot as plt
import time
from utils import project_one_chain, flight_phase
import config


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





n_clusters = 3

points = np.random.normal(size=(1000, 2))
path_R = r['matrix'](FloatVector(np.array(list(points.T.flatten()))), nrow=points.shape[0])

resKMeans = r["kmeans"](path_R, n_clusters, 30)
cluster = dict(zip(resKMeans.names, list(resKMeans)))["cluster"]
cluster = np.array([r for r in cluster])
centroids = dict(zip(resKMeans.names, list(resKMeans)))["centers"]
centroids = np.array([r for r in centroids])
centroids = centroids.reshape(-1, n_clusters, order='C').T

dist, radiuses = compute_centroids_distances(centroids, points, cluster)
new_radiuses = regulate_radius(dist, radiuses)

print("Rad")
print(radiuses)
print(new_radiuses)

plt.scatter(points[:, 0], points[:, 1], c=cluster, marker=".")
plt.scatter(centroids[:, 0], centroids[:, 1], c=["g", "g", "g"], marker = "*")
circle1 = plt.Circle(centroids[0, :], radiuses[0], alpha = 0.1)
circle2 = plt.Circle(centroids[1, :], radiuses[1], alpha = 0.1)
circle3 = plt.Circle(centroids[2, :], radiuses[2], alpha = 0.1)

circle1_new = plt.Circle(centroids[0, :], new_radiuses[0], alpha = 0.1)
circle2_new = plt.Circle(centroids[1, :], new_radiuses[1], alpha = 0.1)
plt.gca().add_patch(circle1)
plt.gca().add_patch(circle2)
plt.gca().add_patch(circle3)
#plt.gca().add_patch(circle1_new)
#plt.gca().add_patch(circle2_new)
plt.show()
d = (np.sqrt(np.sum((centroids[0, :] - points)**2, axis = 1)))
r = np.max(d[np.array(cluster) == 1])

