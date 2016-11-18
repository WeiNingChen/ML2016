import pickle
import os
import sys
import numpy as np
import pandas as pd
from keras.models import load_model
from self_train import load_data, process_unlabel_data


# input image dimensions
img_rows, img_cols = 32, 32
img_channels = 3

# File Path
'''test_path = '~/Desktop/data/test.p'
model_path = sys.argv[1]
output_path = 'temp.csv'
'''
test_path = sys.argv[1]+'test.p'
model_path = sys.argv[2]
output_path = sys.argv[3]



if __name__ == '__main__':

  # Load test data, cnn model
  print 'Loading test data and model...'
  test_data = load_data(test_path)
  X_test = process_unlabel_data(test_data['data'])
  encoder = load_model(model_path+'_ac')
  model = pickle.load(open(model_path+'_svm'))

  # Predict test label
  print 'Predict test labels...'
  y_test = model.predict(encoder.predict(X_test).reshape(X_test.shape[0],128))
  
  # Write output file
  #ids = [i+1  for i in range(len(y_test))]
  print 'Write to the file...'
  output = pd.DataFrame({'ID': test_data['ID'], 'class': y_test})
  output.to_csv(output_path, index = False)

