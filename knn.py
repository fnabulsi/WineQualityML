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
k_range = range(1,3)#
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
y_pred = model.predict(X_test)
conf_matrix, accuracy, recall_array, precision_array = func_confusion_matrix(y_test, y_pred)

print(accuracy)


'''
# 6. Apply the Model to the Test Data
prediction = knn.predict(X_test)


# 7. Display the Results


#print ('\nAccuracy Score: %.3f' % accuracy_score(Y_test, prediction)

#print ('\n******************', \
#'\nConfusion Matrix', \
#'\n*******************'
#print confusion_matrix(Y_test, prediction)


# find the most suitable K
# k = 1  accuracy = 0.584
# k = 2  accuracy = 0.531
# k = 3  accuracy = 0.522
# k = 4  accuracy = 0.550
# k = 5  accuracy = 0.525
'''
'''
for i in range(20):
    k = i + 1
    knn = KNeighborsClassifier(n_neighbors=k)
    #knn.fit(traindata, traintarget)
    knn.fit(X_train,Y_train)
    
    # 6. Apply the Model to the Test Data
    prediction = knn.predict(X_test)
    print 'When k = %d, Accuracy Score: %.3f' % (k,accuracy_score(Y_test, prediction))

# result
When k = 1, Accuracy Score: 0.584
When k = 2, Accuracy Score: 0.531
When k = 3, Accuracy Score: 0.522
When k = 4, Accuracy Score: 0.550
When k = 5, Accuracy Score: 0.525
When k = 6, Accuracy Score: 0.500
When k = 7, Accuracy Score: 0.500
When k = 8, Accuracy Score: 0.497
When k = 9, Accuracy Score: 0.487
When k = 10, Accuracy Score: 0.506
When k = 11, Accuracy Score: 0.522
When k = 12, Accuracy Score: 0.516
When k = 13, Accuracy Score: 0.553
When k = 14, Accuracy Score: 0.544
When k = 15, Accuracy Score: 0.541
When k = 16, Accuracy Score: 0.528
When k = 17, Accuracy Score: 0.531
When k = 18, Accuracy Score: 0.528
When k = 19, Accuracy Score: 0.537
When k = 20, Accuracy Score: 0.531
'''








