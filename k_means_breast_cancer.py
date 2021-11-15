import numpy as np
import matplotlib as mpl
from numpy.core.numeric import ones
from numpy.lib.function_base import _calculate_shapes
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import time as tt

global raw_data
raw_data = load_breast_cancer()

class Centroid(object):
    def __init__(self, idx=None, vec=None, *args, **kwargs):
        self.index = idx
        self.vec = vec

    def update_vec(self, vec):
        self.vec = vec

class DataPoint(object):
    def __init__(self, idx=None, vec=None, assignment=None, *args, **kwargs):
        self.index = idx
        self.vec = vec
        self.assignment = assignment

    def update_assign(self, assignment):
        self.assignment = assignment

def massage_data(dataset, *args, **kwargs):
    '''
        Take all 30 feature datapoints or each of 569 dataset instances, 

        skl dataset obj -> [DataPoint obj]
    '''
    data = []
    for i in range((len(dataset.data))):
        data.append(DataPoint(i, dataset.data[i][0:30]))
    
    return(data)

def k_init(dataset, k_num, *args, **kwargs):
    '''
        Computes minimum and maximum worst values to generate bounds for k initializations to ensure
        centroids are initialized within bounds of dataset. Returns k_num centroid vectors within bounds.

        skl dataset obj, int -> [Centroid obj]
    '''

    centroids = []
    rands = []

    for i in range(k_num):
        centroids.append([])
    
    for i in range(len(centroids)):
        rand_k = np.random.randint(0, len(dataset.data))
        if rand_k not in rands:
            rands.append(rand_k)
            for val in dataset.data[rand_k]:
                centroids[i].append(val)
    
    for i in range(len(centroids)):
        centroids[i] = Centroid(i, centroids[i])

    '''
        Deprecated code using only means:

        # max_vals = [[np.inf, 0], [np.inf, 0], [np.inf, 0], [np.inf, 0], [np.inf, 0], 
        #             [np.inf, 0], [np.inf, 0], [np.inf, 0], [np.inf, 0], [np.inf, 0]]

        # for i in range(len(dataset.data)):
        #     for j in range(10):
        #         if dataset.data[i][j+20] > max_vals[j][1]:
        #             max_vals[j][1] = dataset.data[i][j+20]
        #         elif dataset.data[i][j+20] < max_vals[j][0]:
        #             max_vals[j][0] = dataset.data[i][j+20]
        
        # for i in range(k_num):
        #     centroids.append([])
        #     for j in range(10):
        #         centroids[i].append(np.random.uniform(max_vals[j][0], max_vals[j][1]))
        #     centroids[i] = Centroid(i, centroids[i])
    '''
    
    return (centroids)

def l2_norm(point_vec, centroid_vec=None, *args, **kwargs):
    '''
        Computes distance between two vectors as L2 norm

        [float], [float] -> float
    '''
    dist = 0
    size = len(point_vec)
    if size != len(centroid_vec):
        return ('vector size mismatch')
    else:
        for i in range(len(point_vec)):
            dist += (abs(point_vec[i]) - abs(centroid_vec[i]))**2

    return (dist)

def assign_points(points, centroids, *args, **kwargs):
    '''
        Assigns each point to nearest centroid

        [DataPoint obj], [Centroid] -> [DataPoint objs]
    '''
    for point in iter(points):
        dist = np.inf
        for centroid in iter(centroids):
            if (type(dist) != float and type(dist) != np.float64) or (type(l2_norm(point.vec, centroid.vec)) != np.float64):
                print (type(l2_norm(point.vec, centroid.vec)), type(dist))
            if l2_norm(point.vec, centroid.vec) < dist:
                dist = l2_norm(point.vec, centroid.vec)
                point.update_assign(centroid.index)

    return (points)

def update_centroids(points, centroids, *args, **kwargs):
    '''
        Updates centroid vectors after assigning all points to nearest centroid

        [DataPoint obj], [Centroid] -> [Centroid objs]
    '''
    for j in range(len(centroids)):
        new_vec = []
        point_count = 0
        for i in range(len(centroids[j].vec)):
            new_vec.append(0)
        for point in iter(points):
            if point.assignment == j:
                point_count += 1
                for k in range(len(point.vec)):
                    new_vec[k] += point.vec[k]
        if point_count != 0:
            for l in range(len(new_vec)):
                new_vec[l] = new_vec[l]/point_count
            centroids[j].update_vec(new_vec)
        
    return (centroids)

def k_means(dataset, k_num, *args, **kwargs):
    '''
        The big boy function

        skl dataset obj, int -> ([Centroid objs], [DataPoint objs])
    '''
    points = massage_data(dataset)
    centroids = k_init(dataset, k_num)
    dist_change = 1

    while dist_change != 0:
        
        comparison = []
        for i in range(len(centroids)):
            comparison.append([])
            for j in range(len(centroids[i].vec)):
                comparison[i].append(centroids[i].vec[j])

        assign_points(points, centroids)
        update_centroids(points, centroids)

        dist_change = len(centroids)*len(centroids[0].vec)
        for i in range(len(centroids)):
            for j in range(len(centroids[i].vec)):
                if comparison[i][j] == centroids[i].vec[j]:
                    dist_change -= 1

    return(centroids, points)

def cost_optimize(k_means_out, *args, **kwargs):
    '''
        Calculates cost value for each k-value

        [([Centroid objs], [DataPoint objs])] -> ([int], [float])
    '''
    cost_func = []
    k_vals = []

    for i in range(len(k_means_out[0][0]), len(k_means_out[-1][0]) + 1):
        k_vals.append(i)
                
    for pair in k_means_out:
        cost = 0
        for i in range(len(pair[0])):
            for j in range(len(pair[1])):
                if pair[0][i].index == pair[1][j].assignment:
                    cost += (l2_norm(pair[0][i].vec, pair[1][j].vec))
        cost_func.append(cost/len(pair[1]))  
    
    return (k_vals, cost_func)

def runtime(start):
    '''
        Self-explanatory
    '''
    end_time = tt.time()
    return (end_time-start)

def main_func():
    start = tt.time()

    output = []
    for i in range(2, 4):
        print('Running case for k = {}'.format(i))
        output.append(k_means(raw_data, i))

    plot = cost_optimize(output)
    plt.plot(plot[0], plot[1], '--.')
    plt.xlabel('Number of Centroids, k')
    plt.ylabel('Distortion, J')
    plt.title('K-Means Algorithm Distortion vs. K-Value')

    print('Total Runtime: {}'.format(runtime(start)))
    plt.show()
    

main_func()
