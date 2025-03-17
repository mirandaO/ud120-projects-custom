#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
t0 = time()
classifier.fit(features_train, labels_train)
print("training time:", round(time()-t0, 3), "s")
accuracy = classifier.score(features_test, labels_test)
print(accuracy)


##This number is the one that solves the excersice after submitting it in the Course Quiz.
truncated_accuracy = int(accuracy*10000) / 10000.0
if (truncated_accuracy) == 0.9732:
    print("Exercise solved")

print("####################This is for understanding the features_train contents##########################")
print(str(features_train.shape) + "Where: (n, k) -> n rows, k columns")
print("Data type of elements: " + str(features_train.dtype))
print("Number of dimensions " + str(features_train.ndim))
print("Total number of elements: " + str(features_train.size))
print("features_train: \n" + str(features_train))
print("features_train[10000, 3000]: " + str(features_train[10000, 3784]))
print("####################END##########################")


print("####################This is for understanding the features_test contents##########################")
print(str(features_test.shape) + "Where: (n, k) -> n rows, k columns")
print("Data type of elements: " + str(features_test.dtype))
print("Number of dimensions " + str(features_test.ndim))
print("Total number of elements: " + str(features_test.size))
print("features_train: \n" + str(features_test))
print("features_train[1000, 3784]: " + str(features_test[1000, 3784]))
print("####################END##########################")

t0 = time()
classifier.predict(features_test)
print("predict time:", round(time()-t0, 3), "s")

#########################################################


