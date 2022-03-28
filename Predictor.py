from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import word_tokenize
stemer = LancasterStemmer()
import numpy as np
import nltk 
import random

class predict():
    def __init__(self, model,words,classes,document, intents):
        self.model = model
        self.classes = classes
        self.words = words
        self.document = document
        self.intents = intents

    def pred(self,sent):
        bag = [0]*len(self.words)
        for word in sent:
            for m in range(0,len(self.words)):
                if word == self.words[m]:
                    bag[m] = 1
        bag = np.array([np.array(x) for x in bag])
        bag = np.expand_dims(bag,0)
        return bag

    def response(self, sentence):
        sent = word_tokenize(sentence)
        for no in range(0,len(sent)):
            sent[no] = stemer.stem(sent[no].lower())
        bag = self.pred(sent)
        results = self.model.predict(bag)
        results = np.argmax(results)
        results  = self.classes[results]
        for intent in self.intents['intents']:
            if intent['tag'] == results:
                return random.choice(intent['responses'])
        


    def ask(self):
        while True:
            input_data = input("User: ")
            reply = self.response(input_data)
            print("Bot: ",reply)