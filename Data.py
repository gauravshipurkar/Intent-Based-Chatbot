#Importing the data 
from nltk import word_tokenize
import json
from nltk.util import re_show
import numpy as np
from pandas import DataFrame, get_dummies


words = []
document = []
classes = []
extreme = [" ? "," ' ", " ! "]
class data():
    def dataset():
        with open('Intents.json') as json_data:
            intents = json.load(json_data)

        for intent in intents['intents']:
            for word in intent['patterns']:
                words.extend(word_tokenize(word))
                document.append([word,intent['tag']])
                if intent['tag'] not in classes:
                    classes.append(intent['tag'])

        return words, document, classes, intents
    
    def preprocessing(words):
        from nltk.stem.lancaster import LancasterStemmer
        stemer = LancasterStemmer()
        for no in range(0,len(words)):
            word = words[no]
            if word not in extreme:
                words[no] = stemer.stem(words[no].lower())

        words = list(set(words))
        return words

    def train_ready(words,classes):
        from nltk.stem.lancaster import LancasterStemmer
        stemer = LancasterStemmer()
        training=[]
        for type in document:
            bag=[]
            row=[]
            word_net = []
            word_net.append(type[0])
            for word in word_net:
                word_net = word_tokenize(word)
            for word in range(0,len(word_net)):
                word_net[word] = stemer.stem(word_net[word].lower())
            for w in words:
                if w in word_net:
                    bag.append(1)
                else:
                    bag.append(0)

            for m in classes:
                if m == type[1]:
                    row.append(1)
                else:
                    row.append(0)
                

            training.append([bag, row])

        return (training)




