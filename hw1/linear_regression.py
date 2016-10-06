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
      trainSet.append([tmp,rawData[hour][9]])
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
    

if __name__== '__main__':
  trainSet = train_data_parser("data/train.csv")
  testSet = test_data_parser("data/test_X.csv")
  print trainSet[0][0][0:10]
  print testSet[0][0][0:10]
