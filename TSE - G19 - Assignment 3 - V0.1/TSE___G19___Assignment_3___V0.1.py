#Modules
import csv

#Function to create a list of reviews and a list of sentiments (+ve/-ve)
def listCreation():
    #Global list defined
    global dataset
    dataset = []
    #Opens the csv file
    fileName = open('IMDB Dataset.csv', 'r', encoding='utf-8')
    file = csv.DictReader(fileName)
    #Iterates each row then appends dataset
    for col in file:
        dataset.append(col['review'])
        dataset.append(col['sentiment'])
listCreation()
