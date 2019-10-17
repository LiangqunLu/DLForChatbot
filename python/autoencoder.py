
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import keras
from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.datasets import mnist
import numpy as np

#######minist data for a simple AE example
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

### AE with regularizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import InputLayer
from keras import regularizers

def AE_simple():
    input_p = 784; output_p = 32
    #encoding_dim = 32
    #seqmodel = Sequential()
    #seqmodel.add(InputLayer(input_shape=(input_p,), name='encoder_input'))
    #seqmodel.add(Dense(output_p, activation='relu'))
    #seqmodel.add(Dense(input_p, activation='sigmoid'))
    inputs = Input(shape=(input_p,))
    encoded = Dense(output_p, activation='relu', activity_regularizer=regularizers.l1(10e-5))(inputs)
    decoded = Dense(input_p, activation='sigmoid')(encoded)
    autoencoder = Model(inputs, decoded)
    return(autoencoder)

model = AE_simple()
model.compile(optimizer='adadelta', loss='binary_crossentropy')
model.fit(x_train, x_train,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))


output = model.predict(x_test)

encoder_layer = Model(model.input, model.get_layer(index = 1)(model.input))
encoder_output = encoder_layer.predict(x_test)


#########convolutional AE
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.callbacks import TensorBoard

###minist preparation for 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1)) 
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

def convAE():
    
    input_img = Input(shape=(28, 28, 1))    
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same', name = "encoder_layer")(x)
    
    # at this point the representation is (4, 4, 8) i.e. 128-dimensional
    
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    autoencoder = Model(input_img, decoded)
    return(autoencoder)
    
#tensorboard --logdir=/tmp/convAE
model = convAE()
model.compile(optimizer='adadelta', loss='binary_crossentropy')
model.fit(x_train, x_train,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/convAE')])

output = model.predict(x_test)    

encoder_layer = Model(model.input, model.get_layer(name = "encoder_layer").output)
encoder_output = encoder_layer.predict(x_test)

###########denoising AE model
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1)) 
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)


def denoising_AE():
    
    input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same', name = "encoder_layer")(x)
    
    # at this point the representation is (7, 7, 32)
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    return(autoencoder)

#tensorboard --logdir=/tmp/denoising_AE
model = denoising_AE()
model.compile(optimizer='adadelta', loss='binary_crossentropy')
model.fit(x_train_noisy, x_train,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test_noisy, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/denoising_AE')])

output = model.predict(x_test)    

encoder_layer = Model(model.input, model.get_layer(name = "encoder_layer").output)
encoder_output = encoder_layer.predict(x_test)

######Sequence-to-sequence autoencoder
from keras.layers import Input, LSTM, RepeatVector, SimpleRNN
from keras.models import Model


def seq2seq_AE():

#https://github.com/keras-team/keras/issues/7231    
#inputs = Input(shape=(timesteps, input_dim))
inp = Input(shape = (28,28))

out = LSTM(units = 200, return_sequences=True, activation='tanh')(inp)
out = LSTM(units = 180, return_sequences=True)(out)
out = LSTM(units = 140, return_sequences=True, activation='tanh')(out)
out = LSTM(units = 120, return_sequences=False, activation='tanh')(out)
#out = SimpleRNN(200, activation='tanh')(inp)

encoded = RepeatVector(28, name = "encoder_layer")(out)

out1 = LSTM(140,return_sequences=True, activation='tanh')(encoded)   
out1 = LSTM(180,return_sequences=True, activation='tanh')(out1)   
out1 = LSTM(200,return_sequences=True, activation='tanh')(out1)   
out1 = LSTM(28,return_sequences=True, activation='sigmoid')(out1) # I also tried softmax instead of sigmoid, not really a difference
#out1 = SimpleRNN(200, activation='tanh')(out)

autoencoder = Model(inp, out1)
autoencoder.summary()

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

#tensorboard --logdir=/tmp/seq2seq_AE
model = autoencoder

model.fit(x_train, x_train,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='./tmp/seq2seq_AE')])

output = model.predict(x_test)    

encoder_layer = Model(model.input, model.get_layer(name = "encoder_layer").output)
encoder_output = encoder_layer.predict(x_test)






