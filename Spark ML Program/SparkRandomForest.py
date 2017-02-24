from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkConf, SparkContext
from numpy import array
import numpy as np

# Boilerplate Spark stuff:
conf = SparkConf().setMaster("spark://master01:7077").setAppName("SparkRandomForest")
sc = SparkContext(conf = conf)


# Convert a list of raw fields from our CSV file to a
# LabeledPoint that MLLib can use. All data must be numerical...
def createLabeledPoints(fields):
    features = array(fields[9:])
    label = int(fields[1])
    return LabeledPoint(label,features)

def createTestPoints(fields):
	features = array(fields[9:])
	return features

#Load up our CSV file
trainRaw = sc.textFile("/Demo/train.csv")
testRaw = sc.textFile("/Demo/test.csv")

# Split each line into a list based on the comma delimiters
csvTrain = trainRaw.map(lambda x: x.encode("ascii", "ignore").split(",") )
csvTest = testRaw.map(lambda x :  x.encode("ascii", "ignore").split(",") )


# Convert these lists to LabeledPoints and Non-LablelPoints
trainData = csvTrain.map(createLabeledPoints)
testData = csvTest.map(createTestPoints)


# Train our RandomForest classifier using our data set
model = RandomForest.trainClassifier(trainData, numClasses=98, categoricalFeaturesInfo={},
                                     numTrees=100, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=10, maxBins=32)


# Now get predictions for our test candidates.
predictions = model.predict(testData)



'''
print ('Predcited value:')
results = predictions.collect()
for result in results:
    print str(result)+'\n'
'''

#wrtie the predicted value with its Id in a file
ids = csvTest.map(lambda x: int(x[0])
resultTuple = ids.zip(predictions)
resultTuple.saveAsTextFile("/Demo/")
