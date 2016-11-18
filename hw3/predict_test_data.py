import pickle
import os
import sys
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.models import load_model
from self_train import load_data, process_unlabel_data


# input image dimensions
img_rows, img_cols = 32, 32
img_channels = 3

# File Path
test_path = '~/Desktop/data/test.p'
model_path = sys.argv[1]
output_path = 'temp.csv'

'''
test_path = sys.argv[1]+'test.p'
model_path = sys.argv[2]
output_path = sys.argv[3]
'''


if __name__ == '__main__':

  # Load test data, cnn model
  print 'Loading test data and model...'
  test_data = load_data(test_path)
  X_test = process_unlabel_data(test_data['data'])
  model = load_model(model_path)

  # Predict test label
  print 'Predict test labels...'
  y_test = model.predict_classes(X_test)
  
  # Write output file
  #ids = [i+1  for i in range(len(y_test))]
  print 'Write to the file...'
  output = pd.DataFrame({'ID': test_data['ID'], 'class': y_test})
  output.to_csv(output_path, index = False)

