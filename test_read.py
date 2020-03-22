from pyspark.sql import SparkSession


spark = SparkSession.builder \
    .appName("dataset_tweets")\
    .master("local[24]")\
    .config("spark.driver.memory", "8G")\
    .config("spark.driver.maxResultSize", "4G") \
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.4")\
    .config("spark.kryoserializer.buffer.max", "500m")\
    .getOrCreate()

# df = sqlContext.read.parquet('dataset_users.parquet')
df = spark.read.parquet('users_combined.parquet')
df.show(20, False)
df.printSchema()


