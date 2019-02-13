# Load the data with np.loadtxt
import numpy as np
R = np.loadtxt("D:/Stanford/cs246/hw2/q4/data/user-shows.txt")
# Compute the R and R transpose
R_T = np.transpose(R)
RR_T = np.dot(R,R_T)
#Compute P and Q
fShows = open("D:/Stanford/cs246/hw2/q4/data/shows.txt","r")
movies = fShows.readlines()
Alex = R[499]
#P
Diagonal = np.diag(RR_T)
P = np.diag(Diagonal)
#Q
R_TR = np.dot(np.transpose(R),R)
DiagonalQ = np.diag(R_TR)
Q = np.diag(DiagonalQ)
# PSqrtInverse
PSqrtInverse = np.diag(1/np.diag(np.sqrt(P)))
GammaU = np.dot(np.dot(np.dot(np.dot(PSqrtInverse,R),R_T),PSqrtInverse),R)
#Users-Users recommendations
AlexU = GammaU[499]
# Top5Index = []

top5 = sorted(enumerate(AlexU[:100]),key = lambda entry : (-entry[1],entry[0]))[:5]
recommended5 = []
for m in top5:
    recommended5.append(movies[m[0]])
print("Users-Users recommendations:")
print(recommended5)


QSqrtInverse = np.diag(1/np.diag(np.sqrt(Q)))
GammaI = np.dot(np.dot(np.dot(np.dot(R,QSqrtInverse),R_T),R),QSqrtInverse)

#Movies-movies recommendations
AlexM = GammaI[499]
# Top5Index = []
top = sorted(enumerate(AlexM[:100]),key = lambda entry : (-entry[1],entry[0]))[:5]
recommended = []
for m in top:
    recommended.append(movies[m[0]])
print("Movies-movies recommendations:")
print(recommended)