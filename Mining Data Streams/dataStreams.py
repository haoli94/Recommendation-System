import numpy as np
import matplotlib.pyplot as plt
import math
sigma = np.exp(-5)
epsilon = np.exp(1) * np.float(1e-4)
p = 123457
n_buckets  = np.int(np.exp(1)/epsilon)
hash_params = []
count_words = 0
errors = []
freqs = []

def hash_fun(hashParams, p, n_buckets, x):
    res = []
    for param in hashParams:
        a, b = param
        y = x % p
        hash_val = (a * y + b) % p
        res.append(hash_val % n_buckets)
    return res

with open("D:/Stanford/cs246/hw4/q4/data/hash_params.txt","r") as params:
    for line in params.readlines():
        a,b = line.split()
        hash_params.append((int(a),int(b)))
matrix = np.zeros((len(hash_params), n_buckets))
with open("D:/Stanford/cs246/hw4/q4/data/words_stream_tiny.txt","r")as stream:
    streams = []
    for line in stream.readlines():
        count_words += 1
        x = int(line.strip())
        hash_vals = hash_fun(hash_params, p, n_buckets, x)
        for i in range(len(hash_params)):
            matrix[i][hash_vals[i]] += 1
with open("D:/Stanford/cs246/hw4/q4/data/counts_tiny.txt","r") as counts:
    for line in counts.readlines():
        num,count = line.split()
        x = int(num)
        count = int(count)
        hash_vals = hash_fun(hash_params, p, n_buckets, x)
        countMin = matrix[0][hash_vals[0]]
        for i in range(1,len(hash_params)):
            if matrix[i][hash_vals[i]] < countMin:
                countMin = matrix[i][hash_vals[i]]
        freqs.append(count/count_words)
        error = (countMin-count)/count
        errors.append(error)

fig=plt.figure(figsize=(12,8))
plt.loglog(freqs,errors,".",color="blue")
plt.title("Log Errors vs Log Frequency")
plt.xlabel("Log Frequency")
plt.ylabel("Log Error")
plt.grid(alpha=0.5,linestyle='-.')
plt.savefig("D:\Stanford\cs246\hw4\q4\data\LogerrorsVSlogFrequency")