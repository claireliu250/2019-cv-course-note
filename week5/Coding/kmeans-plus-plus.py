import math
import random
from sklearn import datasets

def euler_distance(point1: list, point2: list) -> float:
    # 计算2点之间的欧拉距离
    distance = 0.0
    for a, b in zip(point1, point2):
        distance += math.pow(a - b, 2)
    return math.sqrt(distance)

def get_closest_dist(point, centroids):
    # 无穷大
    min_dist = math.inf
    for i, centroid in enumerate(centroids):
        dist = euler_distance(centroid, point)
        if dist < min_dist:
            min_dist = dist
    return min_dist

def kpp_centers(data_set: list, k: int) -> list:
    cluster_centers = []
    cluster_centers.append(random.choice(data_set))
    d = [0 for _ in range(len(data_set))]
    for _ in range(1, k):
        total = 0.0
        for i, point in enumerate(data_set):
            # 与最近一个聚类中心的距离
            d[i] = get_closest_dist(point, cluster_centers)
            total += d[i]
        total *= random.random()
        for i, di in enumerate(d):
            total -= di
            if total > 0:
                continue
            cluster_centers.append(data_set[i])
            break
    return cluster_centers

if __name__ == '__main__':
    iris = datasets.load_iris()
    print(kpp_centers(iris.data, 8))

