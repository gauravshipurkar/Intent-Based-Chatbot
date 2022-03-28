import numpy as np
import tensorflow as tf
from tensorflow import keras 
from Data import data
from Training import train
from Predictor import predict
from tensorflow.keras.models import Sequential, save_model, load_model

def __init__():
    words, document, classes, intents = data.dataset()
    words = data.preprocessing(words)
    training = data.train_ready(words,classes)
    train_x, train_y = train.ready(training)
    model = train.model(train_x, train_y)
    train.compile(model)
    train.start_training(model,train_x,train_y)
    print("Loading the Model......")
    model = load_model('model.h5')
    pre = predict(model,words,classes,document, intents)
    pre.ask()


__init__()


