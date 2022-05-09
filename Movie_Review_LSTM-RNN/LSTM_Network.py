#Modules
import csv
import re

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
