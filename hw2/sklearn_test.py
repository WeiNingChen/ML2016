import numpy as np
import pandas as pd
import math as math  
import sys
import os

from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential
from keras.layers import Dense



def process_data(filename, skiprow=0):
  '''
  Load and process data omtp a list of pandas DataFrame
  each element in the list = one id
  '''
  df = pd.read_csv(filename, header=None, skiprows=skiprow)
  # drop id
  df.drop(0,axis=1,inplace=True)

  return df

def generate_dataset(data):
  '''
  Store training data as numpy matrix
  '''
  data_X = np.zeros(data.shape)
  data_X[:,:-1] = np.array(data.ix[:,0:data.shape[1]-1], dtype=float)
  data_X[:,-1] = np.zeros(data_X.shape[0])+1
  data_y = np.array(data.ix[:,data.shape[1]], dtype=float)
  return [data_X, data_y]

def generate_testdata(data):
  '''
  Store training data as numpy matrix
  '''
  data_X = np.zeros((data.shape[0], data.shape[1]+1))
  data_X[:,:-1] = np.array(data.ix[:,0:data.shape[1]], dtype=float)
  data_X[:,-1] = np.zeros(data_X.shape[0])+1
  return data_X


if __name__ == '__main__':
  # Training data processing
  data = process_data('data/spam_train.csv')
  [train_X, train_y] = generate_dataset(data)
  print train_X[0:10,0:10]
  print train_y[0:10]
  

  # Train the model
  #model = svm.SVC(kernel = 'linear', C=100).fit(train_X[0:3500], train_y[0:3500])
  #model = KNeighborsClassifier(n_neighbors=500).fit(train_X[0:3500], train_y[0:3500])
  #print 'Test scores'+ str(model.score(train_X[3500:4000], train_y[3500:4000]))
  # Validation Part
  #scores = cross_val_score(model, train_X[:], train_y[:], cv = 3)
  #print 'Test scores: '+ str(scores) + '%'

  # create model
  model = Sequential()
  model.add(Dense(len(train_X[0]), input_dim=len(train_X[0]), init='uniform', activation='relu'))
  model.add(Dense(len(train_X[0]), init='uniform', activation='relu'))
  model.add(Dense(1, init='uniform', activation='sigmoid'))
  # Compile model
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  # Fit the model
  model.fit(train_X[0:3500], train_y[0:3500], nb_epoch=150, batch_size=10)
  # evaluate the model
  scores = model.evaluate(train_X[3500:4000], train_y[3500:4000])
  print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
  
  
  # Get test labels
  testData = process_data('data/spam_test.csv')
  test_X = generate_testdata(testData)
   
  labels_raw = model.predict(test_X)
  labels = [int(i) for i in labels_raw]
  ids = [i+1 for i in range(len(labels))]

  output = pd.DataFrame({'id': ids, 'label': labels})
  output.to_csv('SVM_C_1.csv', index = False)
