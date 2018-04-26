import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import linear_model
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

#Normalize all data
scaler = preprocessing.StandardScaler()
scaler.fit(wineFeatures)
wineFeatures = scaler.transform(wineFeatures)

#Split data into training and testing data
featuresTrain, xTest, labelsTrain, yTest = model_selection.train_test_split(
    wineFeatures, wineLabels, test_size=testSize, random_state=seed)

#Split training data into training and validation
xTrain, xValidation, yTrain, yValidation = model_selection.train_test_split(
    featuresTrain, labelsTrain, test_size=testSize, random_state=seed)

#Run the model
MAX_ITERATIONS = 100
logReg = linear_model.LogisticRegression(
    max_iter=MAX_ITERATIONS, C=10, random_state=seed)
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

numDataToDisplay = 5
successResults = []
failedResults = []
testY = pd.Series.tolist(yTest)
for i in range(0, len(yPred)):
    if(yPred[i] == testY[i]):
        successResults.append(scaler.inverse_transform(xTest[i]))
    else:
        failedResults.append(scaler.inverse_transform(xTest[i]))

for i in range(0, len(successResults)):
    print("Successful input #{}: {}".format(i+1, successResults[i]))
    if(i + 1 == numDataToDisplay):
        break

for i in range(0, len(failedResults)):
    print("Failed input #{}: {}".format(i+1, failedResults[i]))
    if(i + 1 == numDataToDisplay):
        break

#Examine how the hyperparameter C affects error
cList = []
for i in range(1,300):
    c = 0.1 * i
    cList.append(c)

errorList = []
for c in cList:
    logReg =linear_model.LogisticRegression(max_iter=MAX_ITERATIONS, C=c, random_state=seed)
    logReg.fit(xTrain, yTrain)
    error = 1 - logReg.score(xValidation, yValidation)
    errorList.append(error)

minError = min(errorList)
minIndex = errorList.index(minError)
minC = cList[minIndex]
print('The minimum error: {} is created by using a C-Value of: {}.'.format(minError, minC))


plt.plot(cList, errorList)
plt.title('C Value vs. Error')
plt.xlabel('C-Value')
plt.ylabel('Error')
plt.annotate(
    'Best C: {} Error: {}'.format(minC,minError), xy=(minC, minError),xytext=(minC+5, minError+0.005),
    arrowprops=dict(facecolor='black', shrink=0.01, width=1, headwidth=5))
plt.show()


#Examine how the hyperparameter MAX_ITERATIONS affects the error
iterationsList = []
for i in range(1,100):
    iteration = 100 * i
    iterationsList.append(iteration)

errorList = []
for iteration in iterationsList:
    logReg =linear_model.LogisticRegression(max_iter=iteration, C=0.2, random_state=seed)
    logReg.fit(xTrain, yTrain)
    error = 1 - logReg.score(xValidation, yValidation)
    errorList.append(error)

minIterError = min(errorList)
minIterIndex = errorList.index(minIterError)
minIter = iterationsList[minIterIndex]
print('The minimum error: {} is created by using a Max Iterations: {}.'.format(minIterError, minIter))

plt.plot(iterationsList, errorList)
plt.title('Max Iterations vs. Error')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.annotate(
    'Best Iterations: {} Error: {}'.format(minIter,minIterError), xy=(minIter, minIterError),xytext=(minC+5, minError+0.005),
    arrowprops=dict(facecolor='black', shrink=0.01, width=1, headwidth=5))
plt.show()



