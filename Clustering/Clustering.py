import re
import sys
from pyspark import SparkConf, SparkContext
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 5:
    print("Usage information: <data.txt> <centroidPath1> <centroidPath2> <outPutPath>")
    exit(0)

conf = SparkConf()
sc = SparkContext(conf=conf)
MAX_ITER = 20
k = 10
# lines = sc.textFile("D:\Stanford\cs246\hw2\q2\data\data.txt")

lines = sc.textFile(sys.argv[1])

centroidPath1 = sys.argv[2]
centroidPath2 = sys.argv[3]
outPutPath = sys.argv[4]

points = lines.map(lambda l: np.array([float(val) for val in (re.split(r'[ ]+', l))]))

c1 = sc.textFile(centroidPath1).map(lambda l: np.array([float(val) for val in (re.split(r'[ ]+', l))])).collect()
c2 = sc.textFile(centroidPath2).map(lambda l: np.array([float(val) for val in (re.split(r'[ ]+', l))])).collect()


# c1 = sc.textFile(centroidPath1).map(lambda l: np.array([float(val) for val in (re.split(r'[ ]+', l))])).collect()
# c2 = sc.textFile(centroidPath2).map(lambda l: np.array([float(val) for val in (re.split(r'[ ]+', l))])).collect()


def L2Dist(u,v):
    return np.linalg.norm(u-v,ord=2)**2

def L1Dist(u,v):
    return np.linalg.norm(u-v,ord=1)


def Assignment(point,c):
    minDist = float("inf")
    assignment = 0
    for index in range(len(c)):
        dist = L2Dist(point,c[index])
        if dist < minDist:
            assignment = index
            minDist = dist
    return (assignment,minDist)


#2a.1
lst = []
for i in range(MAX_ITER):
    assignmentIteration = points.map(lambda point: (Assignment(point,c1)[0], (point, 1, Assignment(point,c1)[1]))).cache()
    compute = assignmentIteration.reduceByKey(lambda v1, v2: (v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]))
    cost = sum(compute.map(lambda cost: cost[1][2]).collect())
    newCentroid = compute.map(lambda centroid: centroid[1][0] / centroid[1][1])
    c1 = newCentroid.collect()
    lst.append(cost)


fig = plt.figure(figsize=(4, 3))
plt.plot(lst)
outPutPath1 = outPutPath + "/" + "l2ErrorC1.png"
# fig.savefig("D:/Stanford/cs246/hw2/q2/C1error.png")
fig.savefig(outPutPath1)


#2a.2
lstFar = []
for i in range(MAX_ITER):
    assignmentIteration = points.map(lambda point: (Assignment(point,c2)[0], (point, 1, Assignment(point,c2)[1]))).cache()
    compute = assignmentIteration.reduceByKey(lambda v1, v2: (v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]))
    cost = sum(compute.map(lambda cost: cost[1][2]).collect())
    newCentroid = compute.map(lambda centroid: centroid[1][0] / centroid[1][1])
    c2 = newCentroid.collect()
    lstFar.append(cost)

fig2 = plt.figure(figsize=(4, 3))
plt.plot(lstFar)
# fig2.savefig("D:/Stanford/cs246/hw2/q2/C2error.png")

outPutPath2 = outPutPath + "/" + "l2ErrorC2.png"
fig.savefig(outPutPath2)


def AssignmentL1(point,c):
    minDist = float("inf")
    assignment = 0
    for index in range(len(c)):
        dist = L1Dist(point,c[index])
        if dist < minDist:
            assignment = index
            minDist = dist
    return (assignment,minDist)



#2b.1



c1 = sc.textFile(centroidPath1).map(lambda l: np.array([float(val) for val in (re.split(r'[ ]+', l))])).collect()
c2 = sc.textFile(centroidPath2).map(lambda l: np.array([float(val) for val in (re.split(r'[ ]+', l))])).collect()

# c1 = sc.textFile(centroidPath1).map(lambda l: np.array([float(val) for val in (re.split(r'[ ]+', l))])).collect()
# c2 = sc.textFile(centroidPath2).map(lambda l: np.array([float(val) for val in (re.split(r'[ ]+', l))])).collect()


lstL1C1 = []
for i in range(MAX_ITER):
    assignmentIteration = points.map(lambda point: (AssignmentL1(point,c1)[0], (point, 1, AssignmentL1(point,c1)[1]))).cache()
    compute = assignmentIteration.reduceByKey(lambda v1, v2: (v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]))
    cost = sum(compute.map(lambda cost: cost[1][2]).collect())
    newCentroid = compute.map(lambda centroid: centroid[1][0] / centroid[1][1])
    c1 = newCentroid.collect()
    lstL1C1.append(cost)


fig = plt.figure(figsize=(6, 4))
plt.plot(lstL1C1)
# fig.savefig("D:/Stanford/cs246/hw2/q2/C1L1Error.png")

outPutPath3 = outPutPath + "/" + "l1ErrorC1.png"
fig.savefig(outPutPath3)


#2b.2


lstL1C2 = []
for i in range(MAX_ITER):
    assignmentIteration = points.map(lambda point: (AssignmentL1(point,c2)[0], (point, 1, AssignmentL1(point,c2)[1]))).cache()
    compute = assignmentIteration.reduceByKey(lambda v1, v2: (v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]))
    cost = sum(compute.map(lambda cost: cost[1][2]).collect())
    newCentroid = compute.map(lambda centroid: centroid[1][0] / centroid[1][1])
    c2 = newCentroid.collect()
    lstL1C2.append(cost)


fig = plt.figure(figsize=(6, 4))
plt.plot(lstL1C2)
# fig.savefig("D:/Stanford/cs246/hw2/q2/C2L1Error.png")

outPutPath4 = outPutPath + "/" + "l1ErrorC2.png"
fig.savefig(outPutPath4)

sc.stop()
