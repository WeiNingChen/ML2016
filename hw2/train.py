import logistic_regression
import sys
import os
import pickle
import numpy as np
import feature as ft
#from matplotlib.pyplot import subplots, show


if __name__ == '__main__':
  # Training data processing
  data = logistic_regression.process_data(sys.argv[1])
  [preTrain_X, train_y] = logistic_regression.generate_dataset(data)
  
  # Feature extraction
  [train_X, reconstMat] = ft.get_train_feature(preTrain_X[:], 100)

  # Initilize the model 
  model = logistic_regression.lr(eta = 0.1, it = 2000, model_init = 0)
  
  # Train the model
  model.fit(train_X[:], train_y[:])
  
  # Save the model
  pickle.dump([model, reconstMat], open(sys.argv[2], 'wb'))
  
  # Predict the model
  test_X = ft.get_test_feature(preTrain_X[:], reconstMat)
  labels = model.predict(test_X)
  
  # Validation section
  print ' '
  print 'Start Validation!!'
  acc = 0
  for i in range(len(labels)):
    if int(labels[i]) == int(train_y[i]):
      acc += 1
  
  print "Accuracy:" + str(float(100*acc/len(labels))) + "%"
  
  # Graphic model
  '''
  fig_2, (ax1_2, ax2_2) = subplots(1, 2)
  for i in range(300):
    if int(train_y[i]) == 0:
      ax1_2.scatter(train_X[i, 0], train_X[i, 1], c = 'r')
    if int(train_y[i]) == 1:
      ax1_2.scatter(train_X[i, 0], train_X[i, 1], c = 'b')
  
  for i in range(len(labels)):
    if int(train_y[i]) == 0:
      ax2_2.scatter(test_X[i, 0], test_X[i, 1], c = 'r')
    if int(train_y[i]) == 1:
      ax2_2.scatter(test_X[i, 0], test_X[i, 1], c = 'b')
  show()
  '''

