from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.layers import Dense, Dropout
from tensorflow import keras 
import tensorflow as tf
import numpy as np
import random

from tensorflow.python.keras.backend import learning_phase
from tensorflow.python.ops.gen_array_ops import shape

class train():
    
    def ready(training):
        random.shuffle(training)
        training = np.array(training)
        print(training[:,0][0])
        train_x = np.array([np.array(x) for x in training[:,0]])
        train_y = np.array([np.array(y) for y in training[:,1]])
        print(train_x.shape)
        return train_x, train_y
    
    def model(train_x, train_y):
        model  = Sequential([
            Dense(32, input_shape=(train_x.shape[1],), activation='relu'),
            Dropout(0.5),
            Dense(64,activation = 'relu'),
            Dropout(0.6),
            Dense(train_y.shape[1], activation = 'softmax'),
        ])

        return model 
    
    def compile(model):
        model.compile(loss = "categorical_crossentropy", optimizer = Adam(learning_rate = 0.01),metrics = ['accuracy'])
    
    def start_training(model, train_x, train_y):
        model.fit(train_x, train_y, epochs=250, batch_size=5, verbose=1)
        model.save('model.h5')


    
