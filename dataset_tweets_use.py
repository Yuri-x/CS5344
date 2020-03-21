from pyspark import SparkConf, SparkContext
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import SparkSession
from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *
from pyspark.ml import feature as spark_ft
from pyspark.ml import Pipeline


spark = SparkSession.builder \
    .appName("dataset_tweets")\
    .master("local[24]")\
    .config("spark.driver.memory", "8G")\
    .config("spark.driver.maxResultSize", "4G") \
    .config("spark.executor.cores", "8") \
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.4")\
    .config("spark.kryoserializer.buffer.max", "500m")\
    .getOrCreate()

df = spark.read \
    .format("jdbc") \
    .option("url", "jdbc:postgresql://localhost:5432/cs5344") \
    .option("dbtable", "dataset_tweets") \
    .option("user", "spark") \
    .option("password", "password") \
    .option("driver", "org.postgresql.Driver") \
    .load()

# df = spark.read.parquet('tweets_cache.parquet').repartition(24)

document = DocumentAssembler()\
    .setInputCol("tweet_text")\
    .setOutputCol("document")

use = UniversalSentenceEncoder.pretrained() \
 .setInputCols(["document"])\
 .setOutputCol("embeddings")

nlp_pipeline = Pipeline(stages=[document, use])
nlp_model = nlp_pipeline.fit(df)
processed = nlp_model.transform(df)
processed = processed.withColumn("embeddings", F.col("embeddings").getItem(0)["embeddings"]) \
    .select('sponsoring_country', 'tweetid', 'userid', 'tweet_text', 'is_validation', 'embeddings')
processed.write.parquet('tweets_use.parquet')
processed.show(10, False)



