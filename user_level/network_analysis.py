import sys
from pyspark.sql import SparkSession
import pyspark.sql.functions as sqlF
from itertools import chain
from pyspark.sql.functions import col, udf
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, MinHashLSH

reload(sys)
sys.setdefaultencoding('utf-8')
# Start a new session
spark = SparkSession.builder.appName("Network Analysis").getOrCreate()
# Read parquet into dataframe
df = spark.read.parquet("users_combined.parquet")

# Split training and validation (test) datasets
dfTrain = df.filter("is_validation = false")
dfTest = df.filter("is_validation = true")

def concat(type):
	def concat_(*args):
		return list(chain.from_iterable((arg if arg else [] for arg in args)))
	return udf(concat_, ArrayType(type))

concat_string_arrays = concat(StringType())

dfTrainTweets = dfTrain.select(col("userid"), 
	col("sponsoring_country").alias("country"),
	concat_string_arrays("hashtags", "urls", "related_tweetids").alias("combined"))

dfTrainRelatedUsers = dfTrain.select(col("userid"),
	col("sponsoring_country").alias("country"),
	col("related_userids"))

dfTestTweets = dfTest.select(col("userid"), 
	concat_string_arrays("hashtags", "urls", "related_tweetids").alias("combined"))

dfTestRelatedUsers = dfTest.select(col("userid"),
	col("related_userids"))

model = Pipeline(stages=[
	HashingTF(inputCol="combined", outputCol="vectors"),
	MinHashLSH(inputCol="vectors", outputCol="lsh")]).fit(dfTrainTweets)

trainTweetsHashed = model.transform(dfTrainTweets)
testTweetsHashed = model.transform(dfTestTweets)

combined = model.stages[-1].approxSimilarityJoin(trainTweetsHashed, testTweetsHashed, 0.9)

combined.write.parquet('combined_hashed.parquet')

model2 = Pipeline(stages=[
	HashingTF(inputCol="related_userids", outputCol="vectors"),
	MinHashLSH(inputCol="vectors", outputCol="lsh")]).fit(dfTrainRelatedUsers)

trainUsersHashed = model2.transform(dfTrainRelatedUsers)
testUsersHashed = model2.transform(dfTestRelatedUsers)

output = model2.stages[-1].approxSimilarityJoin(trainUsersHashed, testUsersHashed, 0.9)

output.write.parquet('related_userids_hashed.parquet')