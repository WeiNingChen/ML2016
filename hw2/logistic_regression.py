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
  #print(df.head())
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
  print data_X.shape
  print data_y.shape
  return [data_X, data_y]

def sigmoid(z):
  if z >= 100:
    return 1
  if z <= -100:
    return 0  
  return 1/(1+math.exp(-z))

def grad_f(dataset, w):
  g = 0
  for x,y in dataset:
    x = np.array(x)
    #print "w*x"
    #print -y*w.T.dot(x)
    #print "sigmoid(w*x)"
    #print sigmoid(-y * w.T.dot(x)) 
    g += (-y+sigmoid(w.T.dot(x)))*x
  return g / len(dataset)

def cross_entropy(dataset,w):
  ce = 0
  e = 1e-100
  for x,y in dataset:
    ce += y*np.log(sigmoid(np.dot(x,w))+e)+(1-y)*np.log(1-sigmoid(np.dot(x,w))+e)
  return ce
  
def logistic(dataset, it=10000): 
  [data_X, data_y]=dataset
  #print data_X.shape
  #print data_y.shape
  w = np.zeros(data_X.shape[1]) 
  eta = 0 
  gd_sum = 0
  for i in range(it):
    gd = grad_f(zip(data_X, data_y),w)
    w = w - eta * gd
    gd_sum = gd_sum+np.dot(gd,gd)
    eta = 1000000/np.sqrt(gd_sum)
    if i%200==0:
      #print "eta"
      #print eta
      #print "w"
      #print w
      print "cross entropy:"
      print cross_entropy(zip(data_X, data_y), w)
      #print 'gd_sum'
      #print np.sqrt(gd_sum)
      print "# "+str(i)+" iterations"
      print "-----------------------"
  return w

def predict(data, models):
  if sigmoid(np.dot(data, models)) > 0.5:
    return 1
  return 0 

def AdaGrad(f, gf, n, dataset, theta,T):
    gd_sq_sum = np.zeros(n, dtype=float)
    eta = 1
    e = 1e-8
    for t in range(1, T):
        g = gf(trainSet, theta)
        gd_sq_sum += g*g
        for i in range(0, n):
            theta[i] -= eta * g[i] / np.sqrt(gd_sq_sum[i] + e)
        grad_norm = np.linalg.norm(gf(trainSet, theta))
        print "Itr = %d" % t
        #print "f(theta) =", f(trainSet, theta)
        #print "norm(grad) =", grad_norm
        if grad_norm < 1e-3:
          return theta
    return theta
  

if __name__ == '__main__':
  data = process_data('data/spam_train.csv')
  [train_X, train_y] = generate_dataset(data)
  print train_X[0]
  print train_y[0]
  models = logistic([train_X[0:2800], train_y[0:2800]])
  print models  
  print np.dot(models,train_X[0])
  labels = []
  results = []
  for i in range(50):
    results.append(sigmoid(np.dot(models,train_X[i])))
    labels.append(predict(train_X[i],models))
  cnt = 0
  for i in range(50):
    if int(labels[i])==int(train_y[i]):
      cnt+=1
  print cnt
  print str(float(cnt)*2)+"%"
  print results
  print labels
  print train_y.astype(int)[0:50]
  
  

