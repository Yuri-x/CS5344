#################################
# Importing modules and dataset #
#################################

# initialize pyspark
import findspark
findspark.init()

# Import relevant packages required
import numpy as np
from itertools import chain
from pyspark import SparkConf, SparkContext
from pyspark.sql.session import SparkSession
import pyspark.sql.functions as f
from pyspark.ml.feature import IndexToString, StringIndexer
from pyspark.ml.classification import LogisticRegression, LinearSVC, OneVsRest, NaiveBayes, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize spark session
sc = SparkContext('local')
spark = SparkSession(sc)

# read in dataframe
df = spark.read.parquet('tweets_tfidf.parquet')[['sponsoring_country', 'tweetid', 'userid', 'is_validation', 'tfidf']]


###########################################
# Processing Data before Machine Learning #
###########################################

# calculating class weights since this is an imbalanced dataset
y_collect = df.select("sponsoring_country").groupBy("sponsoring_country").count().collect()
unique_y = [x["sponsoring_country"] for x in y_collect]
total_y = sum([x["count"] for x in y_collect])
unique_y_count = len(y_collect)
bin_count = [x["count"] for x in y_collect]
class_weights_spark = {i: ii for i, ii in zip(unique_y, total_y / (unique_y_count * np.array(bin_count)))} # print(class_weights_spark) # {0.0: 5.0, 1.0: 0.5555555555555556}
mapping_expr = f.create_map([f.lit(x) for x in chain(*class_weights_spark.items())])
df = df.withColumn("weight", mapping_expr.getItem(f.col("sponsoring_country")))

# numerically labelling the sponsoring country
stringIndexer = StringIndexer(inputCol = "sponsoring_country", outputCol = "label")
model = stringIndexer.fit(df)
df = model.transform(df)
label_decoder = sorted(set([(i[0], i[1]) for i in df.select(df.sponsoring_country, df.label).collect()]), key=lambda x: x[0])
label_decoder

# train/test split
train, test = df[df['is_validation'] == False], df[df['is_validation'] == True]

#######################
# Logistic Regression #
#######################

# instantiate the base classifier.
lr = LogisticRegression(featuresCol='tfidf', weightCol='weight', maxIter=10, tol=1E-6, fitIntercept=True)

# train the multiclass model.
model = lr.fit(train)

# score the model on test data.
predictions = model.transform(test)

# obtain evaluator.
evaluator = MulticlassClassificationEvaluator(metricName="f1")

# compute the classification error on test data.
f1 = evaluator.evaluate(predictions)
print("f1 score = %g" % (f1))

# to show dataframe with predictions and probabilities
#display(predictions)

# to save predictions -- on Azure Databricks
#predictions.write.save("/FileStore/lr_output.parquet")

####################################
# One-vs-All (Logistic Regression) #
####################################

# instantiate the base classifier.
lr = LogisticRegression(featuresCol='tfidf', weightCol='weight', maxIter=10, tol=1E-6, fitIntercept=True)

# instantiate the One Vs Rest Classifier.
ovr = OneVsRest(featuresCol='tfidf', weightCol='weight', classifier=lr)

# train the multiclass model.
ovrModel = ovr.fit(train)

# score the model on test data.
predictions = ovrModel.transform(test)

# obtain evaluator.
evaluator = MulticlassClassificationEvaluator(metricName="f1")

# compute the classification error on test data.
f1 = evaluator.evaluate(predictions)
print("f1 score = %g" % (f1))

# to show dataframe with predictions and probabilities
#display(predictions)

# to save predictions -- on Azure Databricks
#predictions.write.save("/FileStore/ova_lr_output.parquet")

####################
# One-vs-All (SVM) #
####################

# instantiate the base classifier.
lsvc = LinearSVC(featuresCol='tfidf', weightCol='weight', maxIter=10, regParam=0.1)

# instantiate the One Vs Rest Classifier.
ovr = OneVsRest(featuresCol='tfidf', weightCol='weight', classifier=lsvc)

# train the multiclass model.
ovrModel = ovr.fit(train)

# score the model on test data.
predictions = ovrModel.transform(test)

# obtain evaluator.
evaluator = MulticlassClassificationEvaluator(metricName="f1")

# compute the classification error on test data.
f1 = evaluator.evaluate(predictions)
print("f1 score = %g" % (f1))

# to show dataframe with predictions and probabilities
#display(predictions)

# to save predictions -- on Azure Databricks
#predictions.write.save("/FileStore/ova_svm_output.parquet")

###############
# Naive Bayes #
###############

# instantiate the base classifier.
nb = NaiveBayes(featuresCol='tfidf', weightCol='weight')

# train the multiclass model.
model = nb.fit(train)

# score the model on test data.
predictions = model.transform(test)

# obtain evaluator.
evaluator = MulticlassClassificationEvaluator(metricName="f1")

# compute the classification error on test data.
f1 = evaluator.evaluate(predictions)
print("f1 score = %g" % (f1))

# to show dataframe with predictions and probabilities
#display(predictions)

# to save predictions -- on Azure Databricks
#predictions.write.save("/FileStore/nb_output.parquet")

