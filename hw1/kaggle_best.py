import pandas as pd
import numpy as np
import random
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
  
  for hour in range(MONTH*DATE_PER_MONTH*HOUR-9):
     for j in range(9):
       tmp.extend(rawData[hour+j])
     trainSet.append([np.array(tmp),rawData[hour+9][9]])
     tmp = []
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
    #testSet.append([tmp,'id_'+str(data)])
    testSet.append(tmp)
    tmp=[]
  return testSet

def AdaGrad(f, gf, n, trainSet, theta, IT):
    gsqd = np.zeros(n, dtype=float)
    alpha = 0.5
    e = 1e-8
    for t in range(1, IT):
        g = gf(trainSet, theta)
        gsqd += g*g
        for i in range(0, n):
            theta[i] -= alpha * g[i] / np.sqrt(gsqd[i] + e)
        grad_norm = np.linalg.norm(gf(trainSet, theta))
        print "Itr = %d" % t
        print "f(theta) =", f(trainSet, theta)
        print "norm(grad) =", grad_norm
        if grad_norm < 1e-3:
            return theta
    return theta

def f_loss(trainSet,w):
  rnt = 0
  for i in range(len(trainSet)):
    rnt += np.square(int(round(np.inner(trainSet[i][0],w[0:len(w)-1])+w[len(w)-1]))-trainSet[i][1])
  rnt = np.sqrt(rnt/len(trainSet))
  return rnt

def grad_f(trainSet,w):
  rgconst = 1
  rnt = np.zeros(len(w), dtype=float);
  for i in range(len(trainSet)):
    rnt[0:len(w)-1] = np.add(rnt[0:len(w)-1],2*(np.inner(trainSet[i][0],w[0:len(w)-1])+w[len(w)-1]-trainSet[i][1])*trainSet[i][0])
    #bias term
    rnt[len(w)-1] = np.add(rnt[len(w)-1],2*(np.inner(trainSet[i][0],w[0:len(w)-1])+w[len(w)-1]-trainSet[i][1]))
  return  rnt+2*rgconst*w

def getTestLabel(testData, Model):
  lable = np.inner(testData, Model[0:len(Model)-1])+Model[len(Model)-1]
  return int(round(lable))


if __name__== '__main__':
  #parse data
  trainSet = train_data_parser("data/train.csv")
  testSet = test_data_parser("data/test_X.csv")
  
  #training models
  w_1 = AdaGrad(f_loss, grad_f, 163, trainSet[0:1000], np.zeros(163, dtype=float), 100)
  w = AdaGrad(f_loss, grad_f, 163, trainSet, w_1, 250000)
  
  #get test labels
  labels = [getTestLabel(testData, w) for testData in testSet]
  ids = ['id_'+str(i) for i in range(len(labels))]
  
  #save the result
  pd.DataFrame({'id': ids, 'value': labels}).to_csv("test6.csv", index=False)
  
