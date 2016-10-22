import sys
import pandas as pd
import numpy as np


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
  data_X = np.zeros((data.shape[0], data.shape[1]+1))
  data_X[:,:-1] = np.array(data.ix[:,0:data.shape[1]], dtype=float)
  data_X[:,-1] = np.zeros(data_X.shape[0])+1
  return data_X

def sigmoid(z):
  if z >= 100:
    return 1
  if z <= -100:
    return 0  
  return 1/(1+np.exp(-z))

def predict(data, models):
  #print np.dot(data, models)
  print sigmoid(np.dot(data, models)) > 0.5
  if sigmoid(np.dot(data, models)) > 0.5:
    return 1
  return 0 

if __name__ == '__main__':
  models = np.load(file(sys.argv[1]+".npy"))
  print models[0]
  testData = process_data(file(sys.argv[2]))
  test_X = generate_dataset(testData)
  
  labels = [predict(data, models[0]) for data in test_X]
  ids = [i+1  for i in range(len(labels))]

  output = pd.DataFrame({'id': ids, 'label': labels})
  output.to_csv(sys.argv[3], index = False)
  
