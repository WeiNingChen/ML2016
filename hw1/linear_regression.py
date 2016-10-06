import pandas as pd
import numpy as np
import sys

DIM = 18
HOUR = 24*12*20

def parser(filename) :
  "Parse csv as valid data format 18*(24*240)"
  df = pd.read_csv(filename)
  rawData=[]

  for date in range(240):
    for hour in range(24):
      rawData.append(df.ix[date*18:date*18+17,str(hour)].values)
  
  for hour in range(HOUR):
    for data in range(DIM):
      if rawData[hour][data] == 'NR':
        rawData[hour][data] = 0
      else :
        rawData[hour][data] = float(rawData[hour][data])
  

  return trainSet

'''
class Data:
  def __init__(self, data, label):
    self.data = data
    self.label = label
  
  def labeled(self):
    return !(label=='')

  def toVector(self):
    return data
'''

if __name__== '__main__':
  v = parser("data/train.csv")
  np.array(v)
  
