import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import SparkSession


@F.udf(returnType=T.ArrayType(T.StringType()))
def remove_duplicate(string):
    if not string:
        return list()
    ret = []
    if string[-1] == ',':
        string = string[:-1]
    for each in set(string.split(", ")):
        e = each.strip().strip("'")
        if e:
            ret.append(e)
    return ret


spark = SparkSession.builder \
    .appName("dataset_tweets")\
    .master("local[16]")\
    .config("spark.driver.memory", "16G")\
    .config("spark.driver.maxResultSize", "8G") \
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.4")\
    .config("spark.kryoserializer.buffer.max", "500m")\
    .getOrCreate()

df = spark.read \
    .format("jdbc") \
    .option("url", "jdbc:postgresql://localhost:5432/cs5344") \
    .option("dbtable", "dataset_users") \
    .option("user", "spark") \
    .option("password", "password") \
    .option("driver", "org.postgresql.Driver") \
    .load()

df = df.groupBy('is_validation', 'sponsoring_country', 'userid') \
            .agg(
                 F.collect_set('user_profile_url').alias('user_profile_urls'),
                 F.collect_set('in_reply_to_userid').alias('in_reply_to_userids'),
                 F.collect_set('in_reply_to_tweetid').alias('in_reply_to_tweetids'),
                 F.collect_set('quoted_tweet_tweetid').alias('quoted_tweet_tweetid'),
                 F.collect_set('retweet_userid').alias('retweet_userid'),
                 F.collect_set('retweet_tweetid').alias('retweet_tweetid'),
                 remove_duplicate(F.concat_ws(", ", F.collect_set('hashtags'))).alias('hashtags'),
                 remove_duplicate(F.concat_ws(", ", F.collect_set('urls'))).alias('urls'),
                 remove_duplicate(F.concat_ws(", ", F.collect_set('user_mentions'))).alias('user_mentions'),
            )

df.write.parquet('dataset_users.parquet')


