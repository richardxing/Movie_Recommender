from __future__ import print_function
from pyspark.sql import SparkSession 
spark = SparkSession.builder.appName('rec').master("local[4]").config("spark.driver.cores", 1).config('spark.driver.memory','16g').config('spark.executor.memory','16g').getOrCreate()#create the entry point(singleton object) for spark when using DataFrame
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS #alternative least square, good for parallelization
data = spark.read.csv('/home/richard/Datasets/ml-latest/ratings.csv',inferSchema=True,header=True)
hyperchoosing_data, rest = data.randomSplit([0.005, 0.995]) #used for choosing hyper parameters if hyper is True
(hyper_training, hyper_test) = hyperchoosing_data.randomSplit([0.8, 0.2])
hyper = False
if hyper:
    training = hyper_training
    test = hyper_test
else:
    (training, test) = data.randomSplit([0.8, 0.2])
from pyspark.ml import Pipeline #build a generic pipeline to help with model selection using crossvalidation
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
params = {  #'rank':20,
            'maxIter':15, 
            #'regParam':0.1, 
            'numUserBlocks':10, 
            'numItemBlocks':10, 
            'implicitPrefs':False, 
            #'alpha':1.0, 
            'seed':None, 
            'nonnegative':False, 
            'checkpointInterval':10, 
            'intermediateStorageLevel':'MEMORY_AND_DISK', 
            'finalStorageLevel':'MEMORY_AND_DISK', 
            'coldStartStrategy':"drop" 
            }
pipe_als = ALS(**params, userCol = "userId", itemCol = "movieId", ratingCol = "rating")
pipeline = Pipeline(stages=[pipe_als]) #only estimator
# We now treat the Pipeline as an Estimator, wrapping it in a CrossValidator instance.
# This will allow us to jointly choose parameters for all Pipeline stages.
# A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
# We use a ParamGridBuilder to construct a grid of parameters to search over.
# With 2 values for rank and 2 values for regParam and 2 for alpha(see below),
# this grid will have 8 parameter settings for CrossValidator to choose from.
paramGrid = ParamGridBuilder().addGrid(pipe_als.rank, [15,25])\
.addGrid(pipe_als.regParam, [0.1,0.2])\
.addGrid(pipe_als.alpha, [0.1,1]).build()#
evaluator = RegressionEvaluator(metricName = "rmse", labelCol = "rating",\
                                predictionCol = "prediction")
crossval = CrossValidator(estimator = pipeline,
                          estimatorParamMaps = paramGrid,
                          evaluator = evaluator,
                          numFolds=3)  # use 3+ folds in practice

# Run cross-validation, and choose the best set of parameters.
cvModel = crossval.fit(training)

# Make predictions on test documents. cvModel uses the best model found (lrModel).

predictions = cvModel.transform(test)
rmse = evaluator.evaluate(predictions)
tx = open('/home/richard/results/rmse.txt','w')
tx.write("Root-mean-square error = " + str(rmse))
tx.write(str(paramGrid))
tx.close()

#'paramGrid' is a list of Param maps; 'avgMetrics' is a list of metrics. These 2 lists have the same order. way to find the best set of params

bestPipeline = cvModel.bestModel
bestModel = bestPipeline.stages[0]

import json
f = open('/home/richard/results/parameters.txt','w')
json.dump(cvModel.avgMetrics, f)
f.close()
spark.stop()

#Saving the model for future use
#model_path = "file:/home/richard/model/myCollaborativeFilter"
#bestModel.write().overwrite().save(model_path)

#personalized movie recommender
#from pyspark import SparkContext
#from pyspark.mllib.recommendation import MatrixFactorizationModel
#same_model = MatrixFactorizationModel.load(spark.sparkContext, "file:/home/richard/model/myCollaborativeFilter")
#single_user = 
#reccomendations = same_model.transform(single_user)
#reccomendations.orderBy('prediction',ascending=False).show()