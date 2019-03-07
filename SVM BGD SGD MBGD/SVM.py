import numpy as np
import copy
import matplotlib.pyplot as plt
import time
featuresList = []
targetList = []


featurePath = "D:/Stanford/cs246/hw4/q1/data/features.txt"
targetPath = "D:/Stanford/cs246/hw4/q1/data/target.txt"
savePath = "D:/Stanford/cs246/hw4/q1/Error.png"

with open(featurePath,"r") as file:
    features = file.readlines()
    n = len(features)
    d = len(features[0].strip().split(","))

for f in features:
    featuresList.append(np.array([int(i) for i in f.strip().split(",")]))
targets = open(targetPath,"r")
for t in targets.readlines():
    targetList.append(int(t.strip()))
targets.close()


# BGD

step = 0.0000003
epsilon = 0.25
b = 0
k = 0
C = 100
w = np.zeros(d)
fk = [C * n]


t1 = time.time()
while True:
    targets = open(targetPath,"r")
    features = open(featurePath,"r")
    theta = 0
    for i in range(n):
        feature = np.array([int(i) for i in features.readline().strip().split(",")])
        yi = int(targets.readline())
        margin = np.dot(w,feature)+b
        if (yi*margin)>=1:
            continue
        theta += -yi*feature

    w_new = w - step*(w + C * theta)
    theta2 = 0
    targets = open(targetPath,"r")
    features = open(featurePath,"r")
    for j in range(n):
        feature = np.array([int(i) for i in features.readline().strip().split(",")])
        yi = int(targets.readline())
        margin = np.dot(w,feature)+b
        if yi*margin>=1:
            continue
        theta2 -= yi
    b_new = b - step*C*theta2
    b = copy.deepcopy(b_new)
    w = w_new[:]

    fki = 0/5*np.dot(w,np.transpose(w))
    targets = open(targetPath,"r")
    features = open(featurePath,"r")
    for i in range(n):
        feature = np.array([int(i) for i in features.readline().strip().split(",")])
        yi = int(targets.readline())
        margin = np.dot(w,feature)+b
        if yi*margin>=1:
            continue
        fki += C*(1-yi*margin)
    fk.append(fki)
    delta = abs(fk[-1] - fk[k]) / fk[k] * 100
    if k != 0:
        if delta <= epsilon: break
    k += 1

t2 = time.time()
print("The time for BGD is %.4f"%(t2-t1))


# SGD

step2 = 0.0001
epsilon2 = 0.001
C = 100
w2 = np.zeros(d)
fk2 = [C * n]
k2 = 0
b2 = 0
i = 0
deltas = [0]

t3 = time.time()
while True:
    theta = 0
    feature = featuresList[i]
    yi = targetList[i]

    margin = np.dot(w2, feature) + b2

    if (yi * margin) < 1:
        theta += -yi * feature

    w_new = w2 - step2 * (w2 + C * theta)

    theta2 = 0
    margin = np.dot(w2, feature) + b2
    if (yi * margin) < 1:
        theta2 -= yi

    b2_new = b2 - step2 * C * theta2
    w2 = w_new[:]
    b2 = copy.deepcopy(b2_new)

    fki = 0.5 * np.dot(w2, np.transpose(w2))
    for j in range(n):
        feature = featuresList[j]
        yi = targetList[j]
        margin = np.dot(w2, feature) + b2
        if yi * margin >= 1:
            continue
        fki += C * (1 - yi * margin)
    fk2.append(fki)

    delta1 = abs(fk2[-1] - fk2[k2]) / fk2[k2] * 100
    if k2 == 0:
        pass
    elif k2 == 1:
        deltas.append(delta1)
    else:
        delta = deltas[-1] * 0.5 + delta1 * 0.5
        deltas.append(delta)
        if delta <= epsilon2: break
    k2 += 1
    i = (i + 1) % n

t4 = time.time()
print("The time for SGD is %.4f" % (t4 - t3))

# MBGD

step3 = 0.00001
epsilon3 = 0.01
w3 = np.zeros(d)
C = 100
fk3 = [C * n]
k3 = 0
b3 = 0
batchSize = 20
deltas = [0]
start = 0

t5 = time.time()
while True:
    theta = 0
    end = min(n, (start + 1) * batchSize)
    for i in range(start * batchSize, end):
        feature = featuresList[i]
        yi = targetList[i]
        margin = np.dot(w3, feature) + b3
        if (yi * margin) >= 1:
            continue
        theta += -yi * feature
    w_new = w3 - step3 * (w3 + C * theta)
    theta2 = 0
    for i in range(start * batchSize, end):
        feature = featuresList[i]
        yi = targetList[i]
        margin = np.dot(w3, feature) + b3
        if yi * margin >= 1:
            continue
        theta2 -= yi
    b_new = b3 - step3 * C * theta2
    b3 = copy.deepcopy(b_new)
    w3 = w_new[:]

    fki = 0.5 * np.dot(w3, np.transpose(w3))
    for j in range(n):
        feature = featuresList[j]
        yi = targetList[j]
        margin = np.dot(w3, feature) + b3
        if yi * margin >= 1:
            continue
        fki += C * (1 - yi * margin)
    fk3.append(fki)

    delta1 = abs(fk3[-1] - fk3[k3]) / fk3[k3] * 100
    if k3 == 0:
        pass
    #     elif k3 == 1:
    #         deltas.append(delta1)
    else:
        delta = deltas[-1] * 0.5 + delta1 * 0.5
        deltas.append(delta)
        if delta <= epsilon3: break
    k3 += 1
    start = (start + 1) % ((n + batchSize - 1) // batchSize)

t6 = time.time()
print("The time for MBGD is %.4f" % (t6 - t5))


fig = plt.figure(figsize=(10, 6))
plt.plot(fk)
plt.plot(fk2)
plt.plot(fk3)
plt.xlabel("Number of iterations")
plt.ylabel("Costs")
fig.savefig(savePath)