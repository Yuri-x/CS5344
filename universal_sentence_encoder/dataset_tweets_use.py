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
    .config("spark.driver.memory", "16G")\
    .config("spark.driver.maxResultSize", "64G") \
    .config("spark.executor.cores", "8") \
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.4")\
    .config("spark.kryoserializer.buffer.max", "500m")\
    .getOrCreate()

# df = spark.read \
#     .format("jdbc") \
#     .option("url", "jdbc:postgresql://localhost:5432/cs5344") \
#     .option("dbtable", "dataset_tweets") \
#     .option("user", "spark") \
#     .option("password", "password") \
#     .option("driver", "org.postgresql.Driver") \
#     .load()

df = spark.read.parquet('tweets_cache.parquet').repartition(24)

train_set = df.filter(df['is_validation'] == False)
val_set = df.filter(df['is_validation'] == True)

document = DocumentAssembler()\
    .setInputCol("tweet_text")\
    .setOutputCol("document")

use = UniversalSentenceEncoder.pretrained() \
    .setInputCols(["document"])\
    .setOutputCol("sentence_embeddings")

classsifierdl = ClassifierDLApproach()\
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("class")\
    .setLabelColumn("sponsoring_country")\
    .setMaxEpochs(10)\
    .setEnableOutputLogs(True)

pipeline = Pipeline(stages=[document, use, classsifierdl])
nlp_model = pipeline.fit(train_set)
predictions = nlp_model.transform(val_set)
result = predictions.select('sponsoring_country', 'tweetid', 'userid', "class.result")

result.write.parquet('tweets_use_result.parquet')
result.show(10, False)



