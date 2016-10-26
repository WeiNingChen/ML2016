import logistic_regression 
import pandas as pd
import numpy as np
import random as rnd

from sklearn.model_selection import cross_val_score

  

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
  
  # Cross validation
  mod_init = np.load('model/models_12.npy')
  model = logistic_regression.lr(it = 0, eta =0.1, model_init = mod_init)
  scores = cross_val_score(model, train_X, train_y, scoring = 'accuracy', cv = 10)
  print 'Scores:'
  print scores
  print 'Average :'+str(np.mean(scores))
  
