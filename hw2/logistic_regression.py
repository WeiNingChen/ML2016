import numpy as np
import pandas as pd
import math as math


def process_data(filename, skiprow=0):
  '''
  Load and process data omtp a list of pandas DataFrame
  each element in the list = one id
  '''
  df = pd.read_csv(filename, header=None, skiprows=skiprow)
  # drop id
  df.drop(0,axis=1,inplace=True)
  print('Data Loaded, preview:')
  print df.shape

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

def sigmoid(z):
  if z >= 100:
    return 1
  if z <= -100:
    return 0  
  return 1/(1+np.exp(-z))

def grad_cross_entropy(dataset, w):
  [data_X, data_y] = dataset
  g = 0
  for idx in range(len(data_y)):
    x = data_X[idx]
    y = data_y[idx]
    g += (sigmoid(w.T.dot(x))-y)*x
    #print "w*x"
    #print -y*w.T.dot(x)
    #print "sigmoid(w*x)"
    #print sigmoid(-y * w.T.dot(x)) 
  return g / len(data_y)

def cross_entropy(dataset,w):
  ce = 0
  e = 1e-100
  [data_X, data_y] = dataset
  for x,y in zip(data_X, data_y):
    ce += y*np.log(sigmoid(np.dot(x,w))+e)+(1-y)*np.log(1-sigmoid(np.dot(x,w))+e)
  return -1*ce/len(dataset)
  
def ERM_solver(dataset, loss, grad_loss,  eta=0.1, it=15000): 
  [data_X, data_y]=dataset
  w = np.zeros(len(data_X[0])) 
  gd_sum = 1e-10
  for i in range(it):
    gd = grad_loss(dataset, w)
    gd_sum = gd_sum+np.dot(gd,gd)
    w = w - eta/np.sqrt(gd_sum)*gd 
    if i%200==0:
      #print "eta"
      #print eta
      #print "w"
      #print w
      print "cross entropy:"
      print loss(dataset, w)
      print 'gd'
      print np.dot(gd,gd)
      print "# "+str(i)+" iterations"
      print "-----------------------"
  return w

def predict(data, models):
  if sigmoid(np.dot(data, models)) > 0.5:
    return 1
  return 0 
  

if __name__ == '__main__':
  data = process_data('data/spam_train.csv')
  [train_X, train_y] = generate_dataset(data)
  print train_X[0:10,0:10]
  print train_y[0:10]
  
  models = ERM_solver([train_X[0:1500], train_y[0:1500]], cross_entropy, grad_cross_entropy)
  print models  
  print np.dot(models,train_X[0])
  
  
  labels = []
  results = []
  cnt = 0
  for i in range(2500,2600):
    results.append(sigmoid(np.dot(models,train_X[i])))
    labels.append(predict(train_X[i],models))
    if int(predict(train_X[i], models)) == int(train_y[i]):
      cnt += 1
  
  print "Accuracy: " + str(cnt) + "%"
  print "Sigmoid output: " + str(results)
  print "Prediction labels: " + str(labels)
  print "true labels: " + str(train_y.astype(int)[0:50])
  
  

