import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import SparkSession
from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *
from pyspark.ml import feature as spark_ft
from pyspark.ml import Pipeline


@F.udf(returnType=T.ArrayType(T.StringType()))
def remove_url(array):
    return list(filter(lambda t: t != '.' and not t.startswith(('https://t.co/', 'http://t.co/')), array))


spark = SparkSession.builder \
    .appName("dataset_tweets")\
    .master("local[24]")\
    .config("spark.driver.memory", "16G")\
    .config("spark.driver.maxResultSize", "4G") \
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

document_assembler = DocumentAssembler() \
    .setInputCol("tweet_text") \
    .setOutputCol("document")

sentence_detector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence") \
    .setUseAbbreviations(True)

tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token") \
    .setSplitChars(['-', '@', ',', ';', '=', '>', '<']) \
    .setContextChars(['(', ')', '?', '!', '~', '"', '.'])

stemmer = Stemmer() \
    .setInputCols(["token"]) \
    .setOutputCol("stem")

stopWords = spark_ft.StopWordsRemover.loadDefaultStopWords('english')
stopWords.extend(['rt'])
stopwords = StopWordsCleaner() \
    .setInputCols(["stem"]) \
    .setOutputCol("clean_tokens") \
    .setStopWords(stopWords) \
    .setCaseSensitive(False)

normalizer = Normalizer() \
    .setInputCols(["stem"]) \
    .setOutputCol("normalized")

finisher = Finisher() \
    .setInputCols(["clean_tokens"]) \
    .setOutputCols(["ntokens"]) \
    .setOutputAsArray(True) \
    .setCleanAnnotations(True)

nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, stemmer, stopwords, finisher])
nlp_model = nlp_pipeline.fit(df)
processed = nlp_model.transform(df).persist()
processed = processed.withColumn("ntokens", remove_url(F.col("ntokens")))

tf = spark_ft.HashingTF(numFeatures=1 << 16, inputCol='ntokens', outputCol='tf')
idf = spark_ft.IDF(minDocFreq=5, inputCol='tf', outputCol='tfidf')
feature_pipeline = Pipeline(stages=[tf, idf])

feature_model = feature_pipeline.fit(processed)
features = feature_model.transform(processed).persist()
features.show(100, False)
features = features.select('sponsoring_country', 'tweetid', 'userid', 'tweet_text', 'is_validation', 'tfidf')
features.write.parquet('tweets_tfidf.parquet')



