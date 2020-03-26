import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import SparkSession


@F.udf(returnType=T.ArrayType(T.StringType()))
def remove_duplicate(string):
    if not string:
        return list()
    ret = []
    for each in set(x.lower().strip().strip("'") for x in string.split(", ")):
        if each:
            ret.append(each)
    return ret


spark = SparkSession.builder \
    .appName("dataset_tweets")\
    .master("local[16]")\
    .config("spark.driver.memory", "16G") \
    .config("spark.driver.maxResultSize", "8G") \
    .config("spark.jars", r"J:\spark-2.4.5-bin-hadoop2.7\drivers\postgresql-42.2.11.jar") \
    .config("spark.kryoserializer.buffer.max", "500m") \
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
                 remove_duplicate(F.concat_ws(", ", F.collect_set('hashtags'))).alias('hashtags'),
                 remove_duplicate(F.concat_ws(", ", F.concat_ws(", ", F.collect_set('urls')), F.concat_ws(", ", F.collect_set('user_profile_url')))).alias('urls'),
                 remove_duplicate(F.concat_ws(", ", F.concat_ws(", ", F.collect_set('in_reply_to_tweetid')), F.concat_ws(", ", F.collect_set('quoted_tweet_tweetid')), F.concat_ws(", ", F.collect_set('retweet_tweetid')), F.concat_ws(", ", F.collect_set('tweetid')))).alias('related_tweetids'),
                 remove_duplicate(F.concat_ws(", ", F.concat_ws(", ", F.collect_set('user_mentions')), F.concat_ws(", ", F.collect_set('in_reply_to_userid')), F.concat_ws(", ", F.collect_set('retweet_userid')), F.concat_ws(", ", F.collect_set('userid')))).alias('related_userids'),
            )

df.write.parquet('users_combined.parquet')
df.show(10, False)


