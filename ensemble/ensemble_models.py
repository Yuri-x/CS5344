# initialize pyspark
import findspark
findspark.init()

from pyspark import SparkConf, SparkContext
from pyspark.sql.session import SparkSession

# Import relevant packages required
import pandas as pd
import numpy as np

# Initialize spark session
sc = SparkContext('local')
spark = SparkSession(sc)

# read in output files
tweet_path = "C:/Users/lynxx/Documents/Python Scripts/Spark Project/ova_lr_output/"
user_path = "C:/Users/lynxx/Documents/Python Scripts/Spark Project/User level output/user_level_probability.parquet/"

tweet = spark.read.parquet(tweet_path)[['sponsoring_country', 'userid', 'label', 'prediction']].toPandas()
user = spark.read.parquet(user_path).toPandas()

# create final dataframe (userid + actual label) to be merged on
final = tweet[['userid', 'label']].drop_duplicates(['userid']).reset_index(drop = True)

##################################
### Processing user-level data ###
##################################

# user dataset: split probability into indvidual columns
user[['user0','user1','user2','user3']] = pd.DataFrame(user.probability.values.tolist(), index= user.index)
user = user.drop('probability', axis = 1)

###################################
### Processing tweet-level data ###
###################################

# tweet dataset: groupby userid, then create columns with counts of each label, then calculate probabilities
# count number of predictions for each sponsoring country per userid
tweet = tweet[['userid', 'prediction']].pivot_table(index=['userid'], columns='prediction', aggfunc='size', fill_value=0)
tweet = tweet.reset_index().rename_axis(None, axis=1)

# calculate 'probability' for each userid's label
tweet['tweet0'] = tweet[0.0] / (tweet[0.0] + tweet[1.0] + tweet[2.0] + tweet[3.0])
tweet['tweet1'] = tweet[1.0] / (tweet[0.0] + tweet[1.0] + tweet[2.0] + tweet[3.0])
tweet['tweet2'] = tweet[2.0] / (tweet[0.0] + tweet[1.0] + tweet[2.0] + tweet[3.0])
tweet['tweet3'] = tweet[3.0] / (tweet[0.0] + tweet[1.0] + tweet[2.0] + tweet[3.0])

# prepare tweet data for merge
tweet = tweet[['userid', 'tweet0', 'tweet1', 'tweet2', 'tweet3']]

# merge onto final dataframe and compute prediction
final = final.merge(user, how = 'left', on = 'userid').merge(tweet, how = 'left', on = 'userid')
final['prob0'] = (final['user0']*0.5+final['tweet0']*0.5)
final['prob1'] = (final['user1']*0.5+final['tweet1']*0.5)
final['prob2'] = (final['user2']*0.5+final['tweet2']*0.5)
final['prob3'] = (final['user3']*0.5+final['tweet3']*0.5)
final.drop(['user0', 'user1', 'user2', 'user3', 'tweet0', 'tweet1', 'tweet2', 'tweet3'], axis = 1, inplace = True)
final['prediction'] = np.where((final['prob0'] == final[["prob0", "prob1", "prob2", "prob3"]].max(axis=1)), 0.0, 'pending')
final['prediction'] = np.where((final['prediction'] == 'pending') & (final['prob1'] == final[["prob0", "prob1", "prob2", "prob3"]].max(axis=1)), '1.0', final['prediction'])
final['prediction'] = np.where((final['prediction'] == 'pending') & (final['prob2'] == final[["prob0", "prob1", "prob2", "prob3"]].max(axis=1)), '2.0', final['prediction'])
final['prediction'] = np.where((final['prediction'] == 'pending') & (final['prob3'] == final[["prob0", "prob1", "prob2", "prob3"]].max(axis=1)), '3.0', final['prediction'])
final['prediction'] = pd.to_numeric(final['prediction'], downcast="float")
