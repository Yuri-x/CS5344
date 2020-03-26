import sys
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import col, udf
from pyspark.sql.types import *
import numpy as np

reload(sys)
sys.setdefaultencoding('utf-8')

# Start a new session
spark = SparkSession.builder.appName("Classification").getOrCreate()

def init_zero(x):
	return [0.0, 0.0, 0.0, 0.0]
udfZero = udf(lambda x: init_zero(x), ArrayType(DoubleType()))

def add_vectors(v1, v2):
	newVector = []
	for i, val in enumerate(v1):
		newVector.append(v1[i] + v2[i])
	return newVector

def normalise_vectors(kv):
	key = kv[0]
	vec = kv[1]

	russiaTestTotal = 184
	chinaTestTotal = 410
	iranTestTotal = 245
	venezuelaTestTotal = 134
	for i, val in enumerate(vec):
		if i == 0:
			vec[i] = val / russiaTestTotal
		elif i == 1:
			vec[i] = val / chinaTestTotal
		elif i == 2:
			vec[i] = val / iranTestTotal
		else:
			vec[i] = val / venezuelaTestTotal
	return (key, vec)

def softmax(kv):
	key = kv[0]
	vec = kv[1]
	softmaxVal = np.exp(vec) / np.sum(np.exp(vec), axis=0)
	
	# need to transform back to float as numpy is not supported in dataframe
	newVal = []
	for i, val in enumerate(softmaxVal):
		newVal.append(float(val))
	return (key, newVal)

def form_vector(country, similarity):
	if country == 'russia':
		return [similarity, 0.0, 0.0, 0.0]
	if country == 'china':
		return [0.0, similarity, 0.0, 0.0]
	if country == 'iran':
		return [0.0, 0.0, similarity, 0.0, 0.0]
	if country == 'venezuela':
		return [0.0, 0.0, 0.0, similarity]

def get_rdd(filename):

	df = spark.read.parquet(filename)

	udf_func = udf(lambda country, similarity: form_vector(country, similarity), ArrayType(DoubleType()))

	df = df.withColumn("similarity", (1.0 - col("distCol")))
	df = df.drop("distCol")
	df = df.withColumn("userid", col("datasetB").getField("userid"))
	df = df.withColumn("country", col("datasetA").getField("country"))
	df = df.groupBy("userid", "country").agg(F.count("userid").alias("count"), F.sum("similarity").alias("similarity"))
	df = df.withColumn("sim_vector", udf_func(df.country, df.similarity))
	df = df.drop("country", "count", "similarity")
	df.show()

	return df.rdd.map(tuple)

# 1. Get complete set of validation data
dfTest = spark.read.parquet("users_combined.parquet")
dfTest = dfTest.filter("is_validation = true")

dfCompleteUsers = dfTest.drop("is_validation", "sponsoring_country", "hashtags", "urls",
	"related_tweetids", "related_userids")
dfCompleteUsers = dfCompleteUsers.withColumn("sim_vector", udfZero(dfCompleteUsers.userid))

# 2. Get rdds (from tweets and related usersids) pair of userids and similarity vector
rddComplete = dfCompleteUsers.rdd.map(tuple)

rddRelatedUserids = get_rdd("related_userids_hashed.parquet")
rddRelatedTweetInfo = get_rdd("combined_hashed.parquet")
rddComplete = rddComplete.union(rddRelatedUserids)
rddComplete = rddComplete.union(rddRelatedTweetInfo)
rddComplete = rddComplete.reduceByKey(lambda v1, v2: add_vectors(v1, v2))
rddComplete = rddComplete.map(normalise_vectors)
rddComplete = rddComplete.map(softmax)
dfComplete = rddComplete.toDF()
dfComplete = dfComplete.select(col("_1").alias("userid"), col("_2").alias("probability"))

# print("########### COUNT ##########", dfComplete.count())
# dfComplete.show()

dfComplete.write.parquet('user_level_probability.parquet')
