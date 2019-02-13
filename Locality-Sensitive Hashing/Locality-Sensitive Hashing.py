# Authors: Jessica Su, Wanzi Zhou, Pratyaksh Sharma, Dylan Liu, Ansh Shukla

import numpy as np
import random
import time
import pdb
import unittest
from PIL import Image
import collections
import matplotlib.pyplot as plt

# Finds the L1 distance between two vectors
# u and v are 1-dimensional np.array objects
# TODO: Implement this
def l1(u, v):
    return np.sum(np.abs(u-v))

# Loads the data into a np array, where each row corresponds to
# an image patch -- this step is sort of slow.
# Each row in the data is an image, and there are 400 columns.
def load_data(filename):
    return np.genfromtxt(filename, delimiter=',')

# Creates a hash function from a list of dimensions and thresholds.
def create_function(dimensions, thresholds):
    def f(v):
        boolarray = [v[dimensions[i]] >= thresholds[i] for i in range(len(dimensions))]
        return "".join(map(str, map(int, boolarray)))
    return f

# Creates the LSH functions (functions that compute L K-bit hash keys).
# Each function selects k dimensions (i.e. column indices of the image matrix)
# at random, and then chooses a random threshold for each dimension, between 0 and
# 255.  For any image, if its value on a given dimension is greater than or equal to
# the randomly chosen threshold, we set that bit to 1.  Each hash function returns
# a length-k bit string of the form "0101010001101001...", and the L hash functions 
# will produce L such bit strings for each image.
def create_functions(k, L, num_dimensions=400, min_threshold=0, max_threshold=255):
    functions = []
    for i in range(L):
        dimensions = np.random.randint(low = 0,
                                   high = num_dimensions,
                                   size = k)
        thresholds = np.random.randint(low = min_threshold, 
                                   high = max_threshold + 1, 
                                   size = k)

        functions.append(create_function(dimensions, thresholds))
    return functions

# Hashes an individual vector (i.e. image).  This produces an array with L
# entries, where each entry is a string of k bits.
def hash_vector(functions, v):
    return np.array([f(v) for f in functions])
# decorator, f is the function inside create_functions

# Hashes the data in A, where each row is a datapoint, using the L
# functions in "functions."
def hash_data(functions, A):
    return np.array(list(map(lambda v: hash_vector(functions, v), A)))
# each row has L "0101010001101001..."s， which is a bucket
# different rows are different images

# Retrieve all of the points that hash to one of the same buckets 
# as the query point.  Do not do any random sampling (unlike what the first
# part of this problem prescribes).
# Don't retrieve a point if it is the same point as the query point.
def get_candidates(hashed_A, hashed_point, query_index):
    return filter(lambda i: i != query_index and \
        any(hashed_point == hashed_A[i]), range(len(hashed_A)))

# Retrieve all of the points that hash to one of the same buckets,10 "0101010001101001..."s

# Sets up the LSH.  You should try to call this function as few times as 
# possible, since it is expensive.
# A: The dataset.
# Return the LSH functions and hashed data structure.
def lsh_setup(A, k = 24, L = 10):
    functions = create_functions(k = k, L = L)
    hashed_A = hash_data(functions, A)
    return (functions, hashed_A)

# Run the entire LSH algorithm
def lsh_search(A, hashed_A, functions, query_index, num_neighbors = 10):
    hashed_point = hash_vector(functions, A[query_index, :])
    candidate_row_nums = get_candidates(hashed_A, hashed_point, query_index)
    
    distances = map(lambda r: (r, l1(A[r], A[query_index])), candidate_row_nums)
    best_neighbors = sorted(distances, key=lambda t: t[1])[:num_neighbors]

    return [t[0] for t in best_neighbors]

# Plots images at the specified rows and saves them each to files.
def plot(A, row_nums, base_filename):
    for row_num in row_nums:
        patch = np.reshape(A[row_num, :], [20, 20])
        im = Image.fromarray(patch)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save(base_filename + "-" + str(row_num) + ".png")

# Finds the nearest neighbors to a given vector, using linear search.
def linear_search(A, query_index, num_neighbors):
    distances = map(lambda r: (r, l1(A[r], A[query_index])), filter(lambda i: i != query_index, range(len(A))))
    best_neighbors = sorted(distances, key=lambda t: t[1])[:num_neighbors]
    return [t[0] for t in best_neighbors]

# TODO: Write a function that computes the error measure

def error(LSH_dict,LINEAR_dict,A):
    err = 0
    for index in range(99,1000,100):
        err_LSH = 0
        err_LINEAR = 0
        LSH = LSH_dict[index]
        LINEAR = LINEAR_dict[index]
        for j in range(len(LSH)):
            err_LSH += l1(A[LSH[j]], A[index])
            err_LINEAR += l1(A[LINEAR[j]], A[index])
        err += err_LSH/err_LINEAR
    return err/10

# TODO: Solve Problem 4
def problem4():
    A = load_data("D:\Stanford\cs246\hw1\q4\data\patches.csv")
    functions, hashed_A = lsh_setup(A, k=24, L=10)
    best_nbs = []
    LSH_dict = collections.defaultdict(list)
    LINEAR_dict = collections.defaultdict(list)
    t1 = time.time()
    for index in range(99, 1000, 100):
        LSH_dict[index] = lsh_search(A, hashed_A, functions, index, num_neighbors=3)
        best_nbs.append(str(index) + "\t" + ",".join(map(str, LSH_dict[index])) + "\n")
    t2 = time.time()
    with open("D:/Stanford/cs246/hw1/q4/data/top_similarity.txt", "w") as f:
        for nb in best_nbs:
            f.writelines(nb)
    print("The average runtime of LSH is {}".format((t2-t1)/10))
    LSHsimilar_100 = lsh_search(A, hashed_A, functions, 99, 10)
    plot(A, LSHsimilar_100, "LSH")
    linear_neighbors = []
    t3 = time.time()
    for index in range(99, 1000, 100):
        LINEAR_dict[index] = linear_search(A, index, 3)
        linear_neighbors.append(str(index) + "\t" + ",".join(map(str, LINEAR_dict[index])) + "\n")
    t4 = time.time()
    with open("D:/Stanford/cs246/hw1/q4/data/linear_similarity.txt", "w") as f:
        for lnb in linear_neighbors:
            f.writelines(lnb)
    print("The average runtime of Linear Search is {}".format((t4-t3)/10))
    LINEARsimilar_100 = linear_search(A, 99, 10)
    plot(A, LINEARsimilar_100, "LINEAR")
    err = error(LSH_dict, LINEAR_dict,A)
    print(err)
    errL = []
    for L in range(10, 22, 2):
        functions, hashed_A = lsh_setup(A, 24, L)
        LSH_dict = collections.defaultdict(list)
        for index in range(99, 1000, 100):
            LSH_dict[index] = lsh_search(A, hashed_A, functions, index, 3)
        errL.append(error(LSH_dict, LINEAR_dict,A))
    fig = plt.figure(figsize=(4, 3))
    plt.plot(range(10, 22, 2), errL)
    fig.savefig("D:/Stanford/cs246/hw1/q4/data/Lerrors.png")
    errK = []
    for K in range(16, 26, 2):
        functions, hashed_A = lsh_setup(A, K, 10)
        LSH_dict = collections.defaultdict(list)
        for index in range(99, 1000, 100):
            LSH_dict[index] = lsh_search(A, hashed_A, functions, index, 3)
        errK.append(error(LSH_dict, LINEAR_dict,A))
    fig2 = plt.figure(figsize=(4, 3))
    plt.plot(range(16, 26, 2), errK)
    fig2.savefig("D:/Stanford/cs246/hw1/q4/data/Kerrors.png")


#### TESTS #####

class TestLSH(unittest.TestCase):
    def test_l1(self):
        u = np.array([1, 2, 3, 4])
        v = np.array([2, 3, 2, 3])
        self.assertEqual(l1(u, v), 4)

    def test_hash_data(self):
        f1 = lambda v: sum(v)
        f2 = lambda v: sum([x * x for x in v])
        A = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(f1(A[0,:]), 6)
        self.assertEqual(f2(A[0,:]), 14)

        functions = [f1, f2]
        self.assertTrue(np.array_equal(hash_vector(functions, A[0, :]), np.array([6, 14])))
        self.assertTrue(np.array_equal(hash_data(functions, A), np.array([[6, 14], [15, 77]])))

    ### TODO: Write your tests here (they won't be graded, 
    ### but you may find them helpful)


if __name__ == '__main__':
    unittest.main() ### TODO: Uncomment this to run tests
    problem4()

