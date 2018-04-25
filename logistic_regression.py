import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from conf_matrix import func_confusion_matrix

wine = pd.read_csv("/Users/ftnabulsi/cs596/WineQualityML/winequality-red.csv", delimiter=';')

#Get labels and features
wineLabels = wine['quality']    #Ground truth values
wineFeatures = wine.drop('quality', axis=1) #features (X)

testSize = 0.20
seed = 10

#Split data into training and testing data
featuresTrain, xTest, labelsTrain, yTest = model_selection.train_test_split(wineFeatures, wineLabels, test_size=testSize, random_state=seed)

#Normalize the data
normalizer = preprocessing.MinMaxScaler()
featuresTrain = normalizer.fit_transform(featuresTrain)
xTest = normalizer.fit_transform(xTest)

#Split training data into training and validation
xTrain, xValidation, yTrain, yValidation = model_selection.train_test_split(featuresTrain, labelsTrain, test_size=testSize, random_state=seed)

#Run the model
MAX_ITERATIONS = 100
logReg = linear_model.LogisticRegression(max_iter=MAX_ITERATIONS)
logReg.fit(xTrain, yTrain)
error = 1 - logReg.score(xValidation, yValidation)

print("Training set error is:",error)

#Run the model on the test data
yPred = logReg.predict(xTest)
accuracy = accuracy_score(yTest, yPred)
cMatrix = confusion_matrix(yTest, yPred, labels=wineLabels)
#cMatrix, accuracy, recall, precision = func_confusion_matrix(yTest, yPred)
print("Accuracy on test data is:", accuracy)
print("Confusion Matrix: ")
print(cMatrix)

