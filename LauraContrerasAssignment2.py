# Databricks notebook source
# Laura Contreras Assignment 2

# COMMAND ----------

import pyspark.sql.functions as f
from pyspark.ml.feature import VectorAssembler, StringIndexer, Bucketizer
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.types import StructType, StructField, LongType, StringType, FloatType
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import random
import numpy as np

sc = spark.sparkContext

# COMMAND ----------

# Information Regarding the Dataset:
# For this assignment we will be using the Brain Stroke Data Set from Kaggle Datasets. https://www.kaggle.com/datasets/zzettrkalpakbal/full-filled-brain-stroke-dataset?select=full_filled_stroke_data+%281%29.csv

##Attribute Information

# gender: "Male", "Female" or "Other"
# age: age of the patient
# hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
# heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
# ever_married: "No" or "Yes"
# work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
# Residence_type: "Rural" or "Urban"
# avg_glucose_level: average glucose level in blood
# bmi: body mass index
# smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*
# stroke: 1 if the patient had a stroke or 0 if not
#Note: "Unknown" in smoking_status means that the information is unavailable for this patient

# COMMAND ----------

# Stroke Data Frame, need to fix header
strokeSchema = StructType([StructField('gender', StringType(), True), StructField('age', StringType(), True), 
                           StructField('hypertension', LongType(), True), StructField('heart_disease', LongType(), True), 
                           StructField('ever_married', StringType(), True), StructField('work_type', StringType(), True),
                          StructField('Residence_type', StringType(), True), StructField('avg_glucose_level', FloatType(), True), 
                           StructField('bmi', FloatType(), True), StructField('smoking_status', StringType(), True), 
                           StructField('stroke', LongType(), True)])

stroke = spark.read.format("csv").option("header", True).option("ignoreLeadingWhiteSpace", True).schema(strokeSchema).load('dbfs:///FileStore/tables/full_data_stroke.csv')
stroke = stroke.withColumnRenamed("stroke","label").withColumn("age", stroke["age"].cast(LongType()))
stroke.show(10)
stroke.count()

# COMMAND ----------

# We know that smoking_status contains "unknown" values. We will drop the unknown values then split the data set into training and test set
random.seed(58)
stroke = stroke.filter(f.col('smoking_status')!= 'Unknown')
print(f"{stroke.count()} total rows")  # Still have almost 3.5k rows of data dropping the rows with unknown smoking status. 

# Create train/test set with a 30/70 split
tstStroke, trnStroke = stroke.randomSplit([0.3, 0.7])
print(f"{trnStroke.count()} total rows in training dataset")
print(f"{tstStroke.count()} total rows in test dataset")

# COMMAND ----------

#Creating the transformers for age, gender, ever_married, work_type, redicence_type, and smoking_status. Cholesterol and BMI is left as number. 

# Age data
agesplits = [-float("inf"), 10, 20, 30, 40, 50, 60, 70, 80, float("inf")]
ageBucketizer = Bucketizer(splits=agesplits, inputCol="age", outputCol="ageBucket")

# Gender
sexIndexer = StringIndexer(inputCol='gender', outputCol='sexIndex')

# ever_married
marriedIndex = StringIndexer(inputCol='ever_married', outputCol='marriedIndex')

# work_type
workIndex = StringIndexer(inputCol='work_type', outputCol = 'workIndex')

# redicence_type
residenceIndex = StringIndexer(inputCol='Residence_type', outputCol='residenceIndex')

# smoking_status
smokingIndex = StringIndexer(inputCol='smoking_status', outputCol = 'smokingIndex')


# COMMAND ----------

# Building the pipeline for the model
lr = LogisticRegression()

# Building pipeline
vecAssem = VectorAssembler(inputCols=['ageBucket', 'sexIndex', 'marriedIndex', 'workIndex', 'residenceIndex', 
                                      'smokingIndex', 'avg_glucose_level', 'bmi', 'hypertension', 'heart_disease'], outputCol = 'features')
myStages = [ageBucketizer, sexIndexer, marriedIndex, workIndex, residenceIndex, smokingIndex, vecAssem, lr]
p = Pipeline(stages=myStages)

# COMMAND ----------

# Looking at Logistic Model with no changed paramters
pModel = p.fit(trnStroke)
#Testing the model with testing data
Ppreds = pModel.transform(trnStroke)
print("Training Data")
Ppreds.select("label", "probability", "prediction").show(10)

#Evaluting the CV Model, also showing the best parameters by using the estimator parameter map to obtain the best model parameters
acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
print(f"Train Accuracy = {acc.evaluate(Ppreds)}")
print(f"Train F1 = {f1.evaluate(Ppreds)}")

# COMMAND ----------

# Implementing a Cross Validation to select the best model for LR

# Creating the parameter grid for LR model, 
paramGrid = ParamGridBuilder().addGrid(lr.maxIter, [15,20,30,100]).addGrid(lr.regParam, [0, 0.01, 0.1, 0.5]).addGrid(lr.elasticNetParam, [0, 0.1, 0.2, 0.5, 0.8]).build()

#Building Cross-Validation to get the best model
crossval = CrossValidator(estimator = p, estimatorParamMaps=paramGrid, evaluator=f1, numFolds=2)

# Training the model
cvModel = crossval.fit(trnStroke)

#Testing the model with testing data
preds = cvModel.transform(trnStroke)
print("Training Data")
preds.select("label", "probability", "prediction").show(10)

#Evaluting the CV Model, also showing the best parameters by using the estimator parameter map to obtain the best model parameters
print(cvModel.getEstimatorParamMaps()[np.argmax(cvModel.avgMetrics)])
print(f"Best paramters: maxIter = 15, regParam = 0.01, ElasticNet: 0.1")

# COMMAND ----------

# Looking at Logistic Model with updated paramters
Bestlr = LogisticRegression(maxIter=15, regParam=0.01, elasticNetParam=0.1)
myStages = [ageBucketizer, sexIndexer, marriedIndex, workIndex, residenceIndex, smokingIndex, vecAssem, Bestlr]
Bpipe = Pipeline(stages=myStages)
BModel = Bpipe.fit(trnStroke)

#Testing the model with testing data
preds = BModel.transform(trnStroke)
print("Training Data")
preds.select("label", "probability", "prediction").show(10)

#Evaluting the CV Model, also showing the best parameters by using the estimator parameter map to obtain the best model parameters
print(f"Train Accuracy = {acc.evaluate(preds)}")
print(f"Train F1 = {f1.evaluate(preds)}")

# COMMAND ----------

# Streaming for the test data set, using the best model from the CV and the train data set as the historical dataset

# Splitting into partitions
tstStrokerep = tstStroke.repartition(10)

# Creating smaller files from repartitions
dbutils.fs.rm("FileStore/tables/Stroke/", True)
tstStrokerep.write.format("csv").option("header", True).save("FileStore/tables/Stroke/")

# COMMAND ----------

# Source
sourceStream = spark.readStream.format("csv").option("header", True).schema(strokeSchema).option("mode", "dropMalformed").option("maxFilesPerTrigger",1).load("dbfs:///FileStore/tables/Stroke")
strokeStream = sourceStream.withColumnRenamed("stroke","label").withColumn("age", sourceStream["age"].cast(LongType()))

# Query

predsStream = BModel.transform(strokeStream).select("label", "probability", "prediction")

# Sink
sinkStream = predsStream.writeStream.outputMode("append").queryName("testpreds").format("memory").trigger(processingTime='10 seconds').start()

display(predsStream)

# COMMAND ----------

testpredDF = spark.sql("SELECT * FROM testpreds")

# Evaluators for Precision and Recall, F1 and Accuracy already created
precision = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="precisionByLabel")
recall = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="recallByLabel")

print(f"Test Accuracy = {acc.evaluate(testpredDF)}")
print(f"Test F1 = {f1.evaluate(testpredDF)}")
print(f"Test Precision = {precision.evaluate(testpredDF)}")
print(f"Test Recall = {recall.evaluate(testpredDF)}")
