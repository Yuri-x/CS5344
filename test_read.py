from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T
COUNTRIES = ['russia', 'china', 'iran', 'venezuela']


@F.udf(returnType=T.StringType())
def ensemble(user_level, tweet_level):
    weight = (0.5, 0.5)
    argmax = 0
    max_prob = 0
    for i, u, t in zip(range(len(COUNTRIES)), user_level, tweet_level):
        ensemble_prob = u * weight[0] + t * weight[1]
        if ensemble_prob > max_prob:
            max_prob = ensemble_prob
            argmax = i
    return COUNTRIES[argmax]


spark = SparkSession.builder \
    .appName("dataset_tweets")\
    .master("local[24]")\
    .config("spark.driver.memory", "8G")\
    .config("spark.driver.maxResultSize", "4G") \
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.4")\
    .config("spark.kryoserializer.buffer.max", "500m")\
    .getOrCreate()

user_probability = spark.read.parquet('user_level_probability.parquet')
tweet_result = spark.read.parquet('tweets_use_result.parquet')
tweet_probability = tweet_result.groupBy('sponsoring_country', 'userid') \
    .agg(F.count(F.when(F.col('result').getItem(0) == 'russia', True)).alias('russia'),
         F.count(F.when(F.col('result').getItem(0) == 'china', True)).alias('china'),
         F.count(F.when(F.col('result').getItem(0) == 'iran', True)).alias('iran'),
         F.count(F.when(F.col('result').getItem(0) == 'venezuela', True)).alias('venezuela'),
         F.count(F.col('result').getItem(0)).alias('total')) \
    .select('sponsoring_country', 'userid', F.array(F.col('russia') / F.col('total'), F.col('china') / F.col('total'), F.col('iran') / F.col('total'), F.col('venezuela') / F.col('total')).alias('probability'))
result = tweet_probability.join(user_probability, on=['userid'], how='inner') \
    .select(tweet_probability['sponsoring_country'],
            tweet_probability['userid'],
            ensemble(user_probability['probability'], tweet_probability['probability']).alias('result'))
acc = result.where(F.col('sponsoring_country') == F.col('result')).count() * 1.0 / result.count()
print(f"Accuracy: {acc}")
