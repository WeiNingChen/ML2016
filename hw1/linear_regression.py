import pandas as pd
import numpy as np
import sys

DIM = 18
MONTH = 12
DATE_PER_MONTH = 20
HOUR = 24

def float_with_str(str):
  if str =='NR':
    return float(0)
  else:
    return float(str)

def train_data_parser(filename):
  df = pd.read_csv(filename)
  rawData=[]
  trainSet = []
  tmp = []
  # reshape data into 5760(hours)x18(dimensions)
  for date in range(MONTH*DATE_PER_MONTH):
    for hour in range(HOUR):
      rawData.append(df.ix[date*DIM:date*DIM+DIM-1,str(hour)].values)
  # view each 9 hours as a feature vector
  for hour in range(MONTH*DATE_PER_MONTH*HOUR):
    rawData[hour] = [float_with_str(i) for i in rawData[hour]]
    if hour%10 == 9:
      trainSet.append([np.array(tmp),rawData[hour][9]])
      tmp = []
    else: tmp.extend(np.array(rawData[hour]))
  return trainSet

def test_data_parser(filename):
  df = pd.read_csv(filename, header=None).ix[:,2:]
  dataNum = df.shape[0]/DIM
  testSet = []
  currentTestData = []
  tmp=[]
  # reshape each 9 hours data
  for data in range(dataNum):
    currentTestData = np.array(df.ix[DIM*data:DIM*data+DIM-1,2:]).T
    for hour in range(9):
      tmp.extend([float_with_str(i) for i in currentTestData[hour]])
    testSet.append([tmp,'id_'+str(data)])
    tmp=[]
  return testSet

def AdaGrad(f, gf, n, trainSet):
    theta = np.zeros(n, dtype=float)
    bias = 0;
    gsqd = np.zeros(n, dtype=float)
    T = 1000000
    alpha = 1
    e = 1e-8
    for t in range(1, T):
        g = gf(trainSet, theta)
        gsqd += g*g
        for i in range(0, n):
            theta[i] -= alpha * g[i] / np.sqrt(gsqd[i] + e)
        grad_norm = np.linalg.norm(gf(trainSet, theta))
        print "Itr = %d" % t
        #print "theta =", theta
        print "f(theta) =", f(trainSet, theta, bias)
        #print "grad_f(theta) =", gf(trainSet, theta)
        print "norm(grad) =", grad_norm
        if grad_norm < 1e-3:
            return
    pass

def quadratic_loss(trainSet,w):
  rnt = 0
  for i in range(len(trainSet)):
    rnt += np.square(np.inner(trainSet[i][0],w)-trainSet[i][1])
  return rnt

def grad_f(trainSet,w):
  rnt = w-w;
  for i in range(len(trainSet)):
    rnt = np.add(rnt,2*(np.inner(trainSet[i][0],w)-trainSet[i][1])*trainSet[i][0])
  return  rnt

def getTestLabel(testData, Model)
  l = np.inner(testData, Model)
  return l

if __name__== '__main__':
  trainSet = train_data_parser("data/train.csv")
  testSet = test_data_parser("data/test_X.csv")
  w = AdaGrad(quadratic_loss, grad_f, 162, trainSet)
  print w 
