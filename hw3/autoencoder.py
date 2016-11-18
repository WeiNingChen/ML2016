from keras.layers import Input, Dense
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from self_train import load_data, process_labeled_data, process_unlabel_data
from sklearn import svm
import numpy as np
import pickle
import sys

from keras import backend as K
K.set_image_dim_ordering('th')

# Set env path
file_all_label = sys.argv[1]+'all_label.p'
file_all_unlabel = sys.argv[1]+'all_unlabel.p'
file_test = sys.argv[1]+'test.p'
file_model = sys.argv[2]

input_img = Input(shape=(3, 32, 32))

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

# this model maps an input to its encoded representation
encoder = Model(input=input_img, output=encoded)
print encoder.summary()

print 'Loading labeled data...'
label_data = load_data(file_all_label)
print 'Loading unlabeled data...'
unlabel_data = load_data(file_all_unlabel)
print 'Loading test data...'
test_data = load_data(file_test)
   
  
print 'Preprocess labeled data...'
(X_train, y_train) = process_labeled_data(label_data) 
print 'Preprocess test data...'
X_test = process_unlabel_data(test_data['data'])
X_unlabel = process_unlabel_data(unlabel_data)

print X_train.shape
print X_test.shape

autoencoder.fit(X_unlabel, X_unlabel,
                nb_epoch=100,
                batch_size=256,
                shuffle=True,
                validation_data=(X_train, X_train))

X_train_feature = encoder.predict(X_train)
print X_train_feature.shape
X_train_feature = X_train_feature.reshape(X_train.shape[0],128)

encoder.save(file_model+'_ac')

model = svm.SVC(decision_function_shape='ovo', C = 30)
model.fit(X_train_feature, y_train.reshape(y_train.shape[0],))

pickle.dump(model,open(file_model+'_svm', 'wb'))

y_test = model.predict(encoder.predict(X_test).reshape(X_test.shape[0],128))
print y_test[1:100]
