import re
import sys
import itertools
from pyspark import SparkConf, SparkContext

conf = SparkConf()
sc = SparkContext(conf=conf)
# built the environment
lines = sc.textFile(sys.argv[1])
# read in the parameter 1 to open the data
pairs = lines.map(lambda l: (re.split(r'[\t]+', l)))
# basic transformation on the data to split them
alreadyFriends = pairs.flatMap(lambda user: [((each_friend,user[0]),0) if user[0] > each_friend else ((user[0],each_friend),0) for each_friend in user[1].split(',')])
friendConnect = pairs.flatMap(lambda user: [((each_pair[1],each_pair[0]),1) if each_pair[0] > each_pair[1] else ((each_pair[0],each_pair[1]),1) for each_pair in itertools.combinations(user[1].split(','),2)])
# find the pairs that are already friends and possible friends connections
friendConnect.cache()
alreadyFriends.cache()
rrdAll = friendConnect.union(alreadyFriends)
# union combine the two rrd
comm_friend = rrdAll.groupByKey().filter(lambda user: 0 not in user[1])
# remove the pairs which are already friends
comm_friend = comm_friend.map(lambda x: (x[0], sum(x[1])))
# count the number of common friends
rrdRecommend = comm_friend.flatMap(lambda user : [(user[0][0],(user[0][1],user[1])),(user[0][1],(user[0][0],user[1]))])
# construct maps for recommendation rank
user_recommends = rrdRecommend.groupByKey()
# groupby and get a person's possible new friends
user_recommends_top_10 = user_recommends.mapValues(lambda v: sorted(v, key = lambda x:(-x[1],int(x[0])))[:10]).sortBy(lambda x:int(x[0]))

# sort the possible friends by the number of their common friends numbers by descending order and only get the top ten
def outputstring(raw):
    host = raw[0]
    friends = raw[1]
    lst = []
    for friend in friends:
        lst.append(friend[0])
    string = host+"\t"+",".join(lst) + "\r"
    return string
# write a helper function to convert the recommendations into string

user_recommends_output = user_recommends_top_10.map(lambda user: outputstring(user))
user_recommends_output.saveAsTextFile(sys.argv[2])

seperate_users = user_recommends_top_10.filter(lambda x:x[0] in ['924', '8941', '8942', '9019', '9020', '9021', '9022', '9990', '9992', '9993'])
seperate_output = seperate_users.map(lambda user: outputstring(user))
seperate_output_file = sys.argv[2]+"seperate"
seperate_output.saveAsTextFile(seperate_output_file)
# output 10 specific user's friends separately
sc.stop()
