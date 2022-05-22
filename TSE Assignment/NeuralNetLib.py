import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer

import pandas as pd
from sklearn.model_selection import train_test_split

import csv
import re
import pickle

#Function to create a list of reviews and a list of sentiments (+ve/-ve)
def listCreation(inputData, resultData):
    #Opens the csv file
    fileName = open('IMDB Dataset.csv', 'r', encoding='utf-8')
    file = csv.DictReader(fileName)
    #Iterates each row then appends dataset
    for col in file:
        inputData.append(col['review'])
        if col['sentiment'] == 'positive':
          resultData.append(1)
        elif col['sentiment'] == 'negative':
          resultData.append(0)
        else:
          resultData.append("findMe")                          

def dataCleanup(inputData):
    #Iterates through the review
    for i in range(len(inputData)):
      #Replaces any text between <> with a blank
      inputData[i] = re.sub('<.*?>', '', inputData[i])

def tokeniseData(data, Dtype, token):
    if (Dtype == "train"):
        token.fit_on_texts(data)
    data = token.texts_to_sequences(data)
    return data

## Limiting word count and adding padding
def paddingData(data, maxWords):
    for d in range(len(data)): # Loops through data
        if len(data[d]) > maxWords: # If length of data is above max words
            data[d] = data[d][0:120] # Take a slice of the list starting from index 0 to 120

    for d in range(len(data)): # Loop through review
        while len(data[d]) < maxWords: # While the data length is below 120
            data[d].append(0) # Append another 0

    ## Returns the data with padding
    return data



def trainNewModel():
    ## Reading data into list
    ## Splitting data into train and test sets
    inputData = []
    resultData= []

    listCreation(inputData, resultData)
    dataCleanup(inputData)

    ## X is review and y is sent
    ## Split into train and test data
    ## random_state set to 0 for testing
    x_train, x_test, y_train, y_test = train_test_split(inputData, resultData, test_size=0.2, random_state=0) 

    ## Tokenising data
    token = Tokenizer(lower = True) ## Token
        
    x_train = tokeniseData(x_train, "train", token)
    x_test = tokeniseData(x_test, "test", token)

    ## Limiting word count and adding padding
    maxWords = 120 # Sets max number of words
    total_words = len(token.word_index) + 1

    x_test = paddingData(x_test, maxWords)
    x_train = paddingData(x_train, maxWords)

    ## Converting train and test data to tensors
    x_train = tf.convert_to_tensor(x_train, dtype=tf.int32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.int16)
    x_test = tf.convert_to_tensor(x_test, dtype=tf.int32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.int16)



    EMBED_DIM = 32
    LSTM_OUT = 64

    model = keras.Sequential()
    model.add(layers.Embedding(total_words, EMBED_DIM, input_length = maxWords))
    model.add(layers.LSTM(LSTM_OUT, dropout=0.75, recurrent_dropout=0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    print(model.summary())

    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.fit(x_train, y_train, batch_size = 128, epochs = 5, verbose = 1)
    model.evaluate(x_test, y_test, batch_size = 128, verbose = 2)

    model.save('model.h5')
    
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(token, handle, protocol=pickle.HIGHEST_PROTOCOL)



def predictReview(review):
    model = tf.keras.models.load_model('model.h5')
    with open('tokenizer.pickle', 'rb') as handle:
        token = pickle.load(handle)
    maxWords = 120

    review = tokeniseData(review, "test", token)
    review = paddingData(review, maxWords)
    review = tf.convert_to_tensor(review, dtype=tf.int32)

    return model.predict(review)
        
