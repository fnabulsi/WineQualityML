#first read in the data
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn import metrics
from conf_matrix import func_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#the parameter is the directory you store the winequality-red.csv file, make sure use delimiter ;
wine = pd.read_csv("/Users/keepitup/Desktop/winequality-red.csv", delimiter=';')


#second get labels and features
wine_labels = wine['quality'] #the correct label of red wine 1599 labels
wine_features = wine.drop('quality', axis=1) #the features of each wine 1599 rows by 11 columns


#split data into training and testing data
test_size = 0.20 #testing size propotional to wht whole size
seed = 10 #random number, whatever you like
features_train, x_test, labels_train, y_test = model_selection.train_test_split(wine_features, wine_labels,
                                                                                            test_size=test_size, random_state=seed)

x_train, x_validation, y_train, y_validation = model_selection.train_test_split(features_train, labels_train,test_size=test_size, random_state=seed)


# subsets for training models
# subsets for validation
#Fit the KNN Model
k_range = range(1,51)#
KNN_k_error = []
for k_value in k_range:
    clf = KNeighborsClassifier(n_neighbors=k_value)
    clf.fit(x_train,y_train)
    error = 1. - clf.score(x_validation, y_validation)
    KNN_k_error.append(error)
plt.plot(k_range, KNN_k_error)
plt.title('auto KNN')
plt.xlabel('k values')
plt.ylabel('error')
#plt.xticks(c_range)
plt.show()

algorithm_types = ['ball_tree', 'kd_tree', 'brute']
KNN_algorithm_error = []
for algorithm_value in algorithm_types:
    clf = KNeighborsClassifier(algorithm=algorithm_value)
    clf.fit(x_train,y_train)
    error = 1. - clf.score(x_validation, y_validation)
    KNN_algorithm_error.append(error)

plt.plot(algorithm_types, KNN_algorithm_error)
plt.title('SVM by Kernels')
plt.xlabel('Kernel')
plt.ylabel('error')
plt.xticks(algorithm_types)
plt.show()

model = KNeighborsClassifier(n_neighbors=1)
model.fit(x_train,y_train)

## step 5 evaluate your results with the metrics you have developed in HA3,including accuracy, quantize your results. 

success_example=[]
failure_example=[]
y_pred = model.predict(x_test)
conf_matrix, accuracy, recall_array, precision_array = func_confusion_matrix(y_test, y_pred)

print(accuracy)









