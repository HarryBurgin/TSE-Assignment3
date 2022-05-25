#Modules
import csv
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer

#Global list defined
global dataset
dataset = []

#Function to create a list of reviews and a list of sentiments (+ve/-ve)
def listCreation():
    #Opens the csv file
    fileName = open('IMDB Dataset.csv', 'r', encoding='utf-8')
    file = csv.DictReader(fileName)
    #Iterates each row then appends dataset
    for col in file:
        dataset.append([col['review'],col['sentiment']])

def dataCleanup():
    #Iterates through the dataset
    for i in range (len(dataset)):
        for j in range (len(dataset[i])):
            #If current element is a review, not a sentiment
            if j == 0:
                #Replaces any text between <> with a blank
                dataset[i][j] = re.sub('<.*?>', '', dataset[i][j])
        
listCreation()
dataCleanup()

inputData = []
resultData= []

for i in range (len(dataset)):
    inputData.append(dataset[i][0])

for i in range (len(dataset)):
    if dataset[i][1] == 'negative':
        resultData.append(0)
    else:
        resultData.append(1)

x_train, x_test, y_train, y_test = train_test_split(inputData, resultData, test_size=0.20, random_state=0)

#for i in range(len(y_test)):
    #print(y_test[i])

tokenize = Tokenizer(num_words=5000)
#create a dictionary containing words and corresponding unique index
tokenize.fit_on_texts(x_train)

#use the above dictionary to convert the training and test data reviews into indexes
x_train = tokenize.texts_to_sequences(x_train)
x_test = tokenize.texts_to_sequences(x_test)

for i in range(len(x_train)):
    print(x_train[i])
