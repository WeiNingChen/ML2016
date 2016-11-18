from keras.layers import Input, Dense
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from clustering import load_data, process_labeled_data, process_unlabel_data


# File Path
file_all_label = '~/Desktop/data/all_label.p'
file_all_unlabel = '~/Desktop/data/all_unlabel.p'
file_test = '~/Desktop/data/test.p'
file_model = 'model_final.ks'


input_img = Input(shape=(3, 32, 32))
from keras.models import Model

x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(input_img)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
encoded = MaxPooling2D((2, 2), border_mode='same')(x)

# at this point the representation is (8, 4, 4) i.e. 128-dimensional

x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(3, 3, 3, activation='sigmoid', border_mode='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
print autoencoder.summary()

#from keras.datasets import mnist
import numpy as np

# Load data set
print 'Loading labeled data...'
label_data = load_data(file_all_label)
#print 'Loading unlabeled data...'
#unlabel_data = load_data(file_all_unlabel)
print 'Loading test data...'
test_data = load_data(file_test)
   
  
print 'Preprocess labeled data...'
(X_train, y_train) = process_labeled_data(label_data) 
#print 'Preprocess unlabeled data...'
#X_unlabel = process_unlabel_data(unlabel_data)
print 'Preprocess test data...'
X_test = process_unlabel_data(test_data['data'])

#X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
#X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))
print X_train.shape
print X_test.shape

autoencoder.fit(X_train, X_train,
                nb_epoch=50,
                batch_size=256,
                shuffle=True,
                validation_data=(X_test, X_test))
