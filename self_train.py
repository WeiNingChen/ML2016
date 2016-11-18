import pickle
import os
import numpy as np
import sys
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD

#############
from keras import backend as K
K.set_image_dim_ordering('th')
#############

# CNN Training Parameters
batch_size = 32
nb_classes = 10
nb_epoch = 700

# input image dimensions
img_rows, img_cols = 32, 32
img_channels = 3

# File Path
'''
file_all_label = '~/Desktop/data/all_label.p'
file_all_unlabel = '~/Desktop/data/all_unlabel.p'
file_test = '~/Desktop/data/test.p'
file_model = 'model_final.ks'

'''


def load_data(filename):
  '''
  Use pickle to load data
  '''
  return pickle.load(open(os.path.expanduser(filename), 'rb'))

def process_labeled_data(data):
  '''
  Process labeled data with format img[_class][_img]
  '''
  X_train = []
  Y_train = []
  
  for _class in range(len(data)):
    for _img in range(len(data[_class])):
      #X_train.append(np.array(data[_class][_img]).reshape(3,32,32))
      # Need to permutate the channel due to some stupid reason (probably caused by TA...)
      temp = np.array(data[_class][_img]).reshape(3,32,32)
      temp_2 = temp
      temp_2[0] = temp[1]
      temp_2[1] = temp[2]
      temp_2[2] = temp[0]
      X_train.append(temp_2)
      Y_train.append([_class])
  
  X_train = np.array(X_train, dtype = 'uint8')
  Y_train = np.array(Y_train)
  X_train = X_train.astype('float32')
  X_train /= 255
  
  print 'shape of X_train:'
  print X_train.shape
  print 'shape of Y_train:'
  print Y_train.shape
  
  return (X_train, Y_train)

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
  X_new = X_new.astype('float32')
  X_new /= 255
  return X_new
  
def generate_new_model():
  # Initialize the CNN Model
  model = Sequential()
  
  # Max pooling 1
  model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=X_train.shape[1:]))
  model.add(Activation('relu'))
  model.add(Convolution2D(64, 3, 3))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.5))

  # Max pooling 2
  model.add(Convolution2D(128, 3, 3, border_mode='same'))
  model.add(Activation('relu'))
  model.add(Convolution2D(128, 3, 3))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.5))

  # Max pooling 2
  model.add(Convolution2D(256, 3, 3, border_mode='same'))
  model.add(Activation('relu'))
  model.add(Convolution2D(256, 3, 3))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.5))

  model.add(Flatten())
  
  # Fully Cconnected  Layer 1
  model.add(Dense(1024))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
   
  # Fully Cconnected  Layer 1
  model.add(Dense(1024))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
   
  # Fully Cconnected  Layer 1
  model.add(Dense(2048))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  

  model.add(Dense(10))
  model.add(BatchNormalization())
  model.add(Activation('softmax'))

  # Train the model using SGD, momentum
  sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
  model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
  return model

def reload_train_set(X_new, X_train, y_train, model, threshold = 0.6, top_k = 10000):
  '''
  Reload training set by unlabeled data, with scores > threshold
  '''
  # Predict unlabel data
  y_new_label = model.predict_classes(X_new)
  scores = np.max(model.predict(X_new), axis = 1)
  
  # Get top k index
  index_sorted = scores.argsort()
  X_new_topk = X_new[np.array(index_sorted[-1*top_k:])]
  y_new_topk = y_new_label[np.array(index_sorted[-1*top_k:])]
  X_else = X_new[np.array(index_sorted[0:len(X_new)-top_k])]
  
  print 'Reload info:'
  print index_sorted[0:100]
  print scores[0:100]
  print y_new_topk[0:100]

  y_new_topk = np.array(y_new_topk).reshape(len(y_new_topk),1)
  X_train = np.concatenate((X_train, X_new_topk))
  y_train = np.concatenate((y_train, y_new_topk))
  
  return X_else, (X_train, y_train)




if __name__ == '__main__':
  
  
  # Set env path

  file_all_label = sys.argv[1]+'all_label.p'
  file_all_unlabel = sys.argv[1]+'all_unlabel.p'
  file_test = sys.argv[1]+'test.p'
  file_model = sys.argv[2]

  # Load data set
  print 'Loading labeled data...'
  label_data = load_data(file_all_label)
  print 'Loading unlabeled data...'
  unlabel_data = load_data(file_all_unlabel)
  print 'Loading test data...'
  test_data = load_data(file_test)
  #print 'Loading validation data...'
  #(X_train, y_train), (X_val, y_val) = cifar10.load_data()
  
  ''' 
  # Preprocess validation data
  print 'Preprocess validation data...'
  X_val = X_val.astype('float32')
  X_val /= 255
  y_val = np.array(y_val).reshape(len(y_val),1)
  '''

  print 'Preprocess labeled data...'
  (X_train, y_train) = process_labeled_data(label_data) 
  print 'Preprocess unlabeled data...'
  X_unlabel = process_unlabel_data(unlabel_data)
  print 'Preprocess test data...'
  X_test = process_unlabel_data(test_data['data'])
  
  print 'Using unlabeled test data!!'
  X_unlabel = np.concatenate((X_unlabel, X_test))
  
  print 'Generating Model...'
  model = generate_new_model() 
  print model.summary()
  
  # Fit the training set
  print 'Start fitting the model!!'
  model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, shuffle=True)
  print 'Save model...'
  model.save(sys.argv[2])


  # Iteratively reload unlabel data 1 (No Val)
  print 'Reloading training set'
  (X_unlabel, (X_train, y_train)) = reload_train_set(X_unlabel, X_train, y_train, model, top_k = 6000)
 
  print 'The shape of training data:'
  print X_train.shape
  print y_train.shape
  print 'The shape of remaining unlabeled data:'
  print X_unlabel.shape
  model = generate_new_model()
  model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=250, shuffle=True)
  print 'Save model...'
  model.save(sys.argv[2])


  # Iteratively reload unlabel data 2 (No val)
  print 'Reloading training set'
  (X_unlabel, (X_train, y_train)) = reload_train_set(X_unlabel, X_train, y_train, model, top_k = 6000)
  
  print 'The shape of training data:'
  print X_train.shape
  print y_train.shape
  print 'The shape of remaining unlabeled data:'
  print X_unlabel.shape
  model = generate_new_model()
  model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=250, shuffle=True)
  print 'Save model...'
  model.save(sys.argv[2])


  # Iteratively reload unlabel data 3(No val)
  print 'Reloading training set'
  (X_unlabel, (X_train, y_train)) = reload_train_set(X_unlabel, X_train, y_train, model, top_k = 6000)
  
  print 'The shape of training data:'
  print X_train.shape
  print y_train.shape
  print 'The shape of remaining unlabeled data:'
  print X_unlabel.shape
  model = generate_new_model()
  model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=200, shuffle=True)
  print 'Save model...'
  

# Iteratively reload unlabel data 4 (No val)
  print 'Reloading training set'
  (X_unlabel, (X_train, y_train)) = reload_train_set(X_unlabel, X_train, y_train, model, top_k = 6000)
  
  print 'The shape of training data:'
  print X_train.shape
  print y_train.shape
  print 'The shape of remaining unlabeled data:'
  print X_unlabel.shape
  model = generate_new_model()
  model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=200, shuffle=True)
  

# Iteratively reload unlabel data 5(No Val)
  print 'Reloading training set'
  (X_unlabel, (X_train, y_train)) = reload_train_set(X_unlabel, X_train, y_train, model, top_k = 4000)
  
  print 'The shape of training data:'
  print X_train.shape
  print y_train.shape
  print 'The shape of remaining unlabeled data:'
  print X_unlabel.shape
  model = generate_new_model()
  model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=250, shuffle=True)
  model.save(sys.argv[2])


# Iteratively reload unlabel data 6 (No val)
  print 'Reloading training set'
  (X_unlabel, (X_train, y_train)) = reload_train_set(X_unlabel, X_train, y_train, model, top_k = 6000)
  
  print 'The shape of training data:'
  print X_train.shape
  print y_train.shape
  print 'The shape of remaining unlabeled data:'
  print X_unlabel.shape
  model = generate_new_model()
  model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=200, shuffle=True)
  

# Iteratively reload unlabel data 7 (No val)
  print 'Reloading training set'
  (X_unlabel, (X_train, y_train)) = reload_train_set(X_unlabel, X_train, y_train, model, top_k = 6000)
  
  print 'The shape of training data:'
  print X_train.shape
  print y_train.shape
  print 'The shape of remaining unlabeled data:'
  print X_unlabel.shape
  model = generate_new_model()
  model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=200, shuffle=True)


# Iteratively reload unlabel data 8 (No val)
  print 'Reloading training set'
  (X_unlabel, (X_train, y_train)) = reload_train_set(X_unlabel, X_train, y_train, model, top_k = 5000)
  
  print 'The shape of training data:'
  print X_train.shape
  print y_train.shape
  print 'The shape of remaining unlabeled data:'
  print X_unlabel.shape
  model = generate_new_model()
  model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=200, shuffle=True)
  model.save(sys.argv[2])


# Iteratively reload unlabel data 9 (No val)
  print 'Reloading training set'
  (X_unlabel, (X_train, y_train)) = reload_train_set(X_unlabel, X_train, y_train, model, top_k = 10000)
  
  print 'The shape of training data:'
  print X_train.shape
  print y_train.shape
  print 'The shape of remaining unlabeled data:'
  print X_unlabel.shape
  model = generate_new_model()
  model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=250, shuffle=True)
  model.save(sys.argv[2])
