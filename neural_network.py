import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import neural_network
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from conf_matrix import func_confusion_matrix

wine = pd.read_csv(
    "/Users/ftnabulsi/cs596/WineQualityML/winequality-red.csv", delimiter=';')

#Get labels and features
wineLabels = wine['quality']    #Ground truth values
wineFeatures = wine.drop('quality', axis=1) #features (X)

testSize = 0.20
seed = 10

# #Normalize all data
# scaler = preprocessing.StandardScaler()
# scaler.fit(wineFeatures)
# wineFeatures = scaler.transform(wineFeatures)

#Split data into training and testing data
featuresTrain, xTest, labelsTrain, yTest = model_selection.train_test_split(
    wineFeatures, wineLabels, test_size=testSize, random_state=seed)

#Split training data into training and validation
xTrain, xValidation, yTrain, yValidation = model_selection.train_test_split(
    featuresTrain, labelsTrain, test_size=testSize, random_state=seed)

#Run the model
clf = neural_network.MLPClassifier(hidden_layer_sizes=(10,10,10,10,10,10,10,10,10,10), max_iter=500, verbose=10, alpha=0.0001, solver='sgd', tol=0.0000000001)
clf.fit(xTrain, yTrain)
error = 1 - clf.score(xValidation,yValidation)
print("Validation set error: {}".format(error))


yPred = clf.predict(xTest)
accuracy = accuracy_score(yTest, yPred)
print("Test set accuracy: {}".format(accuracy))
cMatrix = confusion_matrix(yPred, yTest)
print(cMatrix)
