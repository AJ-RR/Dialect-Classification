import sklearn
import pandas as pd
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import mixture
from sklearn.preprocessing import StandardScaler
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
import matplotlib.pyplot as plt

DIALECT_DIR = "data"
DIALECT_LIST = [ "IDR1","IDR2","IDR3","IDR4","IDR5","IDR6","IDR7","IDR8","IDR9"]

# Read the dataset
def read_data():

 X = []
 y = []
 data = pd.read_csv("mfcc_features.csv")
 X = data.iloc[:,1:-1]
 y = data.iloc[:,-1]
 return np.array(X), np.array(y)
	
def loop(X, y, dialect_list):
  avg = []
  for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, shuffle = True)
    avg.append(classify(X_train, y_train, X_test, y_test, dialect_list))
  SVM_acc = []
  KNN_acc = []
  LR_acc = []
  for i in  range(100):
    LR_acc.append(avg[i][0])
    KNN_acc.append(avg[i][1])
    SVM_acc.append(avg[1][2])
  
  SVM_avg = sum(SVM_acc)/len(SVM_acc)
  KNN_avg = sum(KNN_acc)/len(KNN_acc)
  LR_avg = sum(LR_acc)/len(LR_acc)
  
  print('SVM Accuracy: ', SVM_avg)

  print('Logistic Regression Accuracy: ', LR_avg)

  print('K-Nearest Neighbour Accuracy', KNN_avg)

def classify(X_train, y_train, X_test, y_test, dialect_list):

	scaler = StandardScaler()
	scaler.fit(X_train)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)
	acc = []
	# Logistic Regression classifier

	logistic_classifier = linear_model.logistic.LogisticRegression()
	logistic_classifier.fit(X_train, y_train)
	logistic_predictions = logistic_classifier.predict(X_test)
	logistic_accuracy = accuracy_score(y_test, logistic_predictions)
	acc.append(logistic_accuracy)
	logistic_cm = confusion_matrix(y_test, logistic_predictions)
	#print("logistic accuracy = " + str(logistic_accuracy))
	#print("logistic_cm:")
	#print(logistic_cm)

	# K-Nearest neighbour classifier

	knn_classifier = KNeighborsClassifier(n_neighbors = 5)
	knn_classifier.fit(X_train, y_train)
	knn_predictions = knn_classifier.predict(X_test)
	knn_accuracy = accuracy_score(y_test, knn_predictions)
	acc.append(knn_accuracy)
	knn_cm = confusion_matrix(y_test, knn_predictions)
	#print("knn accuracy = " + str(knn_accuracy))
	#print("knn_cm:")
	#print(knn_cm)
	#conf_matrix = confusion_matrix(X_test, out_pred)
	#print(conf_matrix)
	# fig = plt.figure()
	# axs = fig.add_subplot(111)
	# caxs = axs.matshow(knn_cm, interpolation='nearest', cmap=plt.cm.Blues)
	# fig.colorbar(caxs)
	# axs.set_xticklabels([""] + DIALECT_LIST)
	# axs.set_yticklabels([""] + DIALECT_LIST)
	# plt.show()

	# SVM

	svm_classifier = svm.SVC(kernel='linear')
	svm_classifier.fit(X_train, y_train)
	svm_predictions = svm_classifier.predict(X_test)
	svm_accuracy = accuracy_score(y_test, svm_predictions)
	acc.append(svm_accuracy)
	svm_cm = confusion_matrix(y_test, svm_predictions)
	
	#print("svm accuracy = " + str(svm_accuracy))
	#print("svm_cm:")
	#print(svm_cm)

	#GMM
	'''gmm_classifier = mixture.GMM(n_components = 9)
	gmm_classifier.fit(X_train, y_train)
	gmm_predictions = gmm_classifier.predict(X_test)
	gmm_accuracy = accuracy_score(y_test, gmm_predictions)
	gmm_cm = confusion_matrix(y_test, gmm_predictions)
	print("gmm accuracy = " + str(gmm_accuracy))
	print("gmm_cm:")
	print(gmm_cm)'''

	return acc


def main():

 base_dir_mfcc = DIALECT_DIR
 dialect_list = [ "IDR1","IDR2","IDR3","IDR4","IDR5","IDR6","IDR7","IDR8","IDR9"]

 X, y = read_data()

 #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, shuffle = True)
 #classify(X_train, y_train, X_test, y_test, dialect_list)
 loop(X, y,dialect_list)


if __name__ == "__main__":

	main()
