import sklearn
import pandas as pd
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import scipy
import os
import sys
import glob
import numpy as np
from sklearn.externals import joblib
from random import shuffle

DIALECT_DIR = "data"
DIALECT_LIST = [ "IDR1","IDR2","IDR3","IDR4","IDR5","IDR6","IDR7","IDR8","IDR9"]

# Read the dataset
def read_data():

 X = []
 y = []
 data = pd.read_csv("dataset_new.csv")
 X = data.iloc[:,1:-1]
 y = data.iloc[:,-1]
 return np.array(X), np.array(y)


def classify(X_train, y_train, X_test, y_test, dialect_list):

	# Logistic Regression classifier

	logistic_classifier = linear_model.logistic.LogisticRegression()
	logistic_classifier.fit(X_train, y_train)
	logistic_predictions = logistic_classifier.predict(X_test)
	logistic_accuracy = accuracy_score(y_test, logistic_predictions)
	logistic_cm = confusion_matrix(y_test, logistic_predictions)
	print("logistic accuracy = " + str(logistic_accuracy))
	print("logistic_cm:")
	print(logistic_cm)

	# K-Nearest neighbour classifier

	knn_classifier = KNeighborsClassifier()
	knn_classifier.fit(X_train, y_train)
	knn_predictions = knn_classifier.predict(X_test)
	knn_accuracy = accuracy_score(y_test, knn_predictions)
	knn_cm = confusion_matrix(y_test, knn_predictions)
	print("knn accuracy = " + str(knn_accuracy))
	print("knn_cm:")
	print(knn_cm)

	# SVM

	svm_classifier = svm.SVC()
	svm_classifier.fit(X_train, y_train)
	svm_predictions = svm_classifier.predict(X_test)
	svm_accuracy = accuracy_score(y_test, svm_predictions)
	svm_cm = confusion_matrix(y_test, svm_predictions)
	print("svm accuracy = " + str(svm_accuracy))
	print("svm_cm:")
	print(svm_cm)




def main():

 base_dir_mfcc = DIALECT_DIR
 dialect_list = [ "IDR1","IDR2","IDR3","IDR4","IDR5","IDR6","IDR7","IDR8","IDR9"]

 X, y = read_data()

 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.10, shuffle = True)
 classify(X_train, y_train, X_test, y_test, dialect_list)



if __name__ == "__main__":

	main()
