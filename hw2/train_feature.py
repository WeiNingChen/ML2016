import logistic_regression
import sys
import os
import pickle
import numpy as np
import feature as ft
from matplotlib.pyplot import subplots, show

if __name__ == '__main__':
  # Training data processing
  data = logistic_regression.process_data(sys.argv[1])
  [preTrain_X, train_y] = logistic_regression.generate_dataset(data)
  print 'Size of preTrain data:'
  print preTrain_X.shape

  temp = ft.quad_mapping(preTrain_X)
  print 'Size of data after poynomial mapping:'
  print temp.shape

  [temp, reconstMat] = ft.pca(temp[:, :-3], 150)
  print 'Size of featured data afte PCA:'
  print temp.shape

  train_X = ft.add_const_column(temp)
  print 'Size of data after add constant column:'
  print train_X.shape
  
  # Train the model
  if os.path.isfile('model/models_12.npy'):
    mdl_init = np.load('model/models_12.npy')
  else :
    mdl_init = 0
  model = logistic_regression.lr(eta = 0.1, it = 60000, model_init = 0)
  
  model.fit(train_X, train_y)
  
  pickle.dump([model, reconstMat], open(sys.argv[2], 'wb'))
 
  test_X = ft.get_test_feature(preTrain_X[3600:4000], reconstMat)
  labels = model.predict(test_X)
  
  # Validation section
  acc = 0
  for i in range(len(labels)):
    if int(labels[i]) == int(train_y[3600+i]):
      acc += 1
  
  print "Accuracy:" + str(float(100*acc/len(labels))) + "%"
  
  # Graphic model
  fig_2, (ax1_2, ax2_2) = subplots(1, 2)
  for i in range(300):
    if int(train_y[i]) == 0:
      ax1_2.scatter(train_X[i, 0], train_X[i, 1], c = 'r')
    if int(train_y[i]) == 1:
      ax1_2.scatter(train_X[i, 0], train_X[i, 1], c = 'b')
  
  for i in range(len(labels)):
    if int(train_y[3600+i]) == 0:
      ax2_2.scatter(test_X[i, 0], test_X[i, 1], c = 'r')
    if int(train_y[3600+i]) == 1:
      ax2_2.scatter(test_X[i, 0], test_X[i, 1], c = 'b')
  show()


