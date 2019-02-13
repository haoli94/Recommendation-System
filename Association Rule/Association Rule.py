import collections
import itertools
with open("D:/Stanford/cs246/hw1/q2/data/browsing.txt","r") as f:
    data = f.readlines()
lst = []
baskets = []
for d in data:
    basket = d.split()
    lst += basket
    baskets.append(basket)
counts = collections.Counter(lst)
length_baskets = len(baskets)
threshold = 100
frequentItems = []
for key,value in counts.items():
    if value >= threshold:
        frequentItems.append(key)
candidatePairs = set()
length = len(frequentItems)
for i in range(length-1):
    for j in range(i+1,length):
        p = (frequentItems[i],frequentItems[j]) if frequentItems[i] < frequentItems[j]\
        else (frequentItems[j],frequentItems[i])
        candidatePairs.add(p)

pairSupports = collections.defaultdict(int)
for basket in baskets:
    for i in range(len(basket)-1):
        for j in range(i+1,len(basket)):
            p = (basket[i],basket[j]) if basket[i] < basket[j]\
            else (basket[j],basket[i])
            if p not in candidatePairs:
                continue
            else:
                pairSupports[p] += 1


frequentPairs = dict()
for pair,supports in pairSupports.items():
    if supports >= threshold:
        frequentPairs[pair] = supports
frequentPairsSet = set(frequentPairs.keys())


assocRules = []
for key,value in frequentPairs.items():
    assocRules.append((key[0],key[1],value/counts[key[0]]))
    assocRules.append((key[1],key[0],value/counts[key[1]]))

assocRules.sort(key = lambda x : (x[2],x[1]),reverse = True)
print(assocRules[:5])


frequentTripplesCandidate = collections.defaultdict(int)
def checkTripples(tripple):
    for i in range(len(tripple)-1):
        for j in range(i+1,len(tripple)):
            p = (tripple[i],tripple[j]) if tripple[i] < tripple[j]\
            else (tripple[j],tripple[i])
            if p not in frequentPairsSet:
                return False
    return True


# for combination in itertools.combinations(sorted(baskets[0]),3):
#     print(combination)
for basket in baskets:
    for combination in itertools.combinations(sorted(basket),3):
        if checkTripples(combination):
            frequentTripplesCandidate[combination] += 1


frequentTripples = dict()
for tripple,supports in frequentTripplesCandidate.items():
    if supports >= threshold:
        frequentTripples[tripple] = supports


assocRules2 = []
for key,value in frequentTripples.items():
    assocRules2.append(((key[1],key[2]),key[0],value/frequentPairs[(key[1],key[2])]))
    assocRules2.append(((key[0],key[2]),key[1],value/frequentPairs[(key[0],key[2])]))
    assocRules2.append(((key[0],key[1]),key[2],value/frequentPairs[(key[0],key[1])]))

assocRules2.sort(key=lambda x: (x[2], x[0]), reverse=True)
assocRules2 = sorted(filter(lambda x:x[2]==1.0,assocRules2))[:5]
print(assocRules2[:5])