import logistic_regression 
import pandas as pd
import numpy as np
import random as rnd

def random_partition(index_list, k=2):
  division = len(lst) / float(k) 
  return [ index_list[int(round(division*i)):int(round(division*(i+1)))] for i in xrange(k) ]
  
'''
def cross_validation(cv, k  , train_X, train_y, loss = cross_entropy, grad_loss = grad_cross_entropy, solver = ERM_solver, models_init = 0, eta = 1, it = 10000):
  # defualt validation iteration = cv
  if k == 0:
    k = cv
  index = range(len(train_y))
  idx_subset = random_partition(index, cv)
  accuracy = []
  for i in range(k):
    # use train[idx_subset[i]] as validation set
    idx_train = []
    # get training index
    for j in range(k):
      if j != i:
        idx_train.extend(idx_subset[j])

    models = ERM_solver([train_X[idx_train], train_y[idx_train]], loss, grad_loss, models_init, eta, it)
    # validate
'''

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

if __name__ == '__main__':
  # Training data processing
  data = process_data('data/spam_train.csv')
  [train_X, train_y] = generate_dataset(data)
  print train_X[0:10,0:10]
  print train_y[0:10]
  
  model = logistic_regression.lr(it = 10000, eta = 0.1)
  model.fit(train_X[0:3500], train_y[0:3500]) 
  labels = model.predict(train_X[3600:3700])
  print labels
  print train_y[0:100]

  acc = 0
  for i in range(100):
    if labels[i] == train_y[i+3600]:
      acc += 1

  print 'Accuracy:'+str(acc)+'%'
