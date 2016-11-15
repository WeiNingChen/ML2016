import pickle
import os
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.models import load_model
from self_train import load_data


###########################
from keras.datasets import cifar10


# input image dimensions
img_rows, img_cols = 32, 32
img_channels = 3

# File Path
test_path = '~/Desktop/data/all_unlabel.p'
model_path = 'model_epoch_1000.ks'
output_path = 'output_epoch_1000.csv'

def process_unlabel_data(unlabel_data):
  '''
  Process unlabel data with format img[_img]
  '''
  X_new = [] 
  for _img in unlabel_data:
    temp = np.array(_img).reshape(3,32,32)
    temp_2 = temp
    temp_2[0] = temp[1]
    temp_2[1] = temp[2]
    temp_2[2] = temp[0]
    X_new.append(temp_2)
  X_new = np.array(X_new)
  return X_new.astype('uint8')

if __name__ == '__main__':

  # Load test data, cnn model
  print 'Loading test data and model...'
  test_data = load_data(test_path)
  X_test = process_unlabel_data(test_data)
  model = load_model(model_path)
 
  print 'Load cifar10 database from keras'
  (X_data_1, y_data_1), (X_data_2, y_data_2) = cifar10.load_data()
  X_data = np.concatenate((X_data_2, X_data_1))
  y_data = np.concatenate((y_data_2, y_data_1))
  
  X_data = np.array(X_data)
  X_data.astype('uint8')

  print 'database loaded!!'
  print 'database shape:'
  print X_data.shape[0]
  print 'Test data shape:'
  print X_test.shape[0]

  X_data.reshape(X_data.shape[0], 3*32*32)
  X_test.reshape(X_test.shape[0], 3*32*32)

 
 
  # Search from the database
  y_test = []
  find = False
  for img in range(X_test.shape[0]):
    for item in range(X_data.shape[0]):
      if np.mean(X_test[img]-X_data[item]) < 5:
        y_test.append(y_data[item])
        print 'Find in database!!'
        print img,item
        print np.amax(X_test[img]-X_data[item])
        find = True
        break
      #print img, item
    y_test.append(-1)
    #print 'Cannot find in database!!'
  
  #ids = [i+1  for i in range(len(y_test))]
  print 'Write to the file...'
  output = pd.DataFrame({'ID': test_data['ID'], 'class': y_test})
  output.to_csv(output_path, index = False)

