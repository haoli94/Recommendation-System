import numpy as np
import matplotlib.pyplot as plt
step = 0.025
lbda = 0.1
k = 20
Max_iter = 40

filePath = "D:/Stanford/cs246/hw2/q3/data/ratings.train.txt"
savePath = "D:/Stanford/cs246/hw2/q3/data/Error.png"
# Define the path that we are going to read data and output the results
P = dict()
Q = dict()
# Initialize P and Q
with open(filePath,"r") as f:
    for l in f.readlines():
        entry = [int(i) for i in l.split()]
        if entry[0] not in Q:
            Q[entry[0]] = np.random.rand(k).reshape(1,k) * np.sqrt(5/k)
        if entry[1] not in P:
            P[entry[1]] = np.random.rand(k).reshape(1,k) * np.sqrt(5/k)


errorLst = []
for i in range(Max_iter):
    error = 0
    ratings = open(filePath, 'r')
    for rate in ratings:
        entry = [int(i) for i in rate.split()]
        q_id = entry[0]
        p_id = entry[1]
        score = entry[2]
        qi = Q[q_id]
        pu = P[p_id]
        pu_T = np.transpose(pu)
        deltaError = 2 * (score - np.dot(qi, pu_T))
        Q[q_id] = qi + step * (deltaError * P[p_id] - 2 * lbda * qi)
        P[p_id] = pu + step * (deltaError * Q[q_id] - 2 * lbda * pu)
        qi_ = Q[q_id]
        pu_ = P[p_id]
    # Update P and Q through each pass
    ratings = open(filePath, 'r')

    for rate in ratings:
        entry = [int(i) for i in rate.split()]
        q_id = entry[0]
        p_id = entry[1]
        score = entry[2]
        qi = Q[q_id]
        pu = P[p_id]
        pu_T = np.transpose(pu)
        error += (score - np.dot(qi, np.transpose(pu))[0][0]) ** 2
    # Compute the errors
    for p_key in P.keys():
        error += np.sum(P[p_key] * P[p_key])
    for q_key in Q.keys():
        error += np.sum(Q[q_key] * Q[q_key])
    # Add up the norm of the P and Q the regularization term
    errorLst.append(error)

fig = plt.figure(figsize=(10, 6))
plt.plot(np.arange(1,Max_iter+1),errorLst)
plt.xlabel("Number of iterations")
plt.ylabel("Errors")
fig.savefig(savePath)
# Plot the error graph and output the result