import logistic_regression
import sys
import os
import pickle
import numpy as np


if __name__ == '__main__':
  # Training data processing
  data = logistic_regression.process_data(sys.argv[1])
  [train_X, train_y] = logistic_regression.generate_dataset(data)
 
  # Train the model
  if os.path.isfile('model/models_12.npy'):
    mdl_init = np.load('model/models_12.npy')
  else :
    mdl_init = 0
  model = logistic_regression.lr(eta = 0.1, it = 1000, model_init = 0)
  model.fit(train_X[0:3500], train_y[0:3500])
  pickle.dump(model, open(sys.argv[2], 'wb'))
  
  labels = model.predict(train_X[3500:3600])
  
  acc = 0
  for i in range(100):
    if int(labels[i]) == int(train_y[3500+i]):
      acc += 1
  print "Accuracy:" + str(acc) + "%"

