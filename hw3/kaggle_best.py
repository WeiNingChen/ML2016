import pickle
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD

###########################
from keras.datasets import cifar10

# CNN Training Parameters
batch_size = 32
nb_classes = 10
nb_epoch = 2000

# input image dimensions
img_rows, img_cols = 32, 32
img_channels = 3

# File Path
file_all_label = '~/Desktop/data/all_label.p'
file_all_unlabel = '~/Desktop/data/all_unlabel.p'

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
  
def reload_train_set(X_new, X_train, y_train, model):
  
  y_new = model.predict_classes(X_new)
  print y_new

  y_new = np.array(y_new).reshape(len(y_new),1)
  X_train = np.concatenate((X_train, X_new))
  y_train = np.concatenate((y_train, y_new))
  
  return (X_train, y_train)

def generate_new_model():
  # Initialize the CNN Model
  model = Sequential()
  
  model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=X_train.shape[1:]))
  model.add(Activation('relu'))
  model.add(Convolution2D(32, 3, 3))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.5))
  
  model.add(Convolution2D(64, 3, 3, border_mode='same'))
  model.add(Activation('relu'))
  model.add(Convolution2D(64, 3, 3))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.5))
  
 
  model.add(Convolution2D(64, 3, 3, border_mode='same'))
  model.add(Activation('relu'))
  model.add(Convolution2D(64, 3, 3))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.5))


  model.add(BatchNormalization())
  model.add(Flatten())
  model.add(Dense(1024))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  
  model.add(Dense(1024))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  
  model.add(Dense(1024))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  
  model.add(Dense(1024))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  
  model.add(Dense(1024))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  
  model.add(Dense(1024))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  
  '''
  model.add(Dense(512))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  '''
  model.add(Dense(nb_classes))
  
  model.add(Activation('softmax'))
  
  # Train the model using SGD, momentum
  sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
  model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
  
  return model



if __name__ == '__main__':
  
  # Load data set
  label_data = load_data(file_all_label)
  
  '''
  unlabel_data = load_data(file_all_unlabel)
  '''
  
  # Validation Set Generation
  (X_train, y_train), (X_test, y_test) = cifar10.load_data()
   
  
  X_test = X_test.astype('float32')
  X_test /= 255
  y_test = np.array(y_test).reshape(len(y_test),1)
  
  ################################
  #X_train = X_train.astype('float32') 
  #X_train = X_train[0:5000]
  #X_train /= 255
  #y_train = np.array(y_train).reshape(len(y_train),1)
  #y_train = y_train[0:5000]
  ##################################

  (X_train, y_train) = process_labeled_data(label_data) 
  
  '''
  X_unlabel = process_unlabel_data(unlabel_data)
  '''

  model = generate_new_model() 
  print model.summary()
  
  # Fit the training set
  model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data = (X_test, y_test), shuffle=True)
  
  
  # Iteratively reload unlabel data
  '''
  iteration = 45
  unlabel_batch_size = int(len(X_unlabel)/iteration)

  for it in range(iteration):
    print 'reload #'+str(it)+' batch'
    (l_num, r_num) = (it*unlabel_batch_size, (it+1)*unlabel_batch_size-1)
    print 'Before reload unlabel data:'
    print X_train.shape
    (X_train, y_train) = reload_train_set(X_unlabel[l_num:r_num], X_train, y_train, model)
    print 'After  reload data:'
    print X_train.shape
    print 'fit current batch...'
    #model = generate_new_model()
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data = (X_test, y_test), shuffle=True)
  '''
  model.save('model_epoch_1000.ks')
