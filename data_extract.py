import csv
import string
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import pickle

"""
Takes a list of statistics about basketball teams and opens the corresponding .csv file
Compiles these files into a single dictionary 
Team name is the key and each value is a dictionary with a key of the statistic and the value
"""
def constructKnowledge(files, year):
    dataSet = dict()
    for name in files:
        csvFile = open('Data/'+year+"/"+name+'.csv', 'rU')
        reader = csv.reader(csvFile)
        for row in reader:
            #If the knowledgebase does not have the current team, create an empty dictionary for it
            if row[0] not in dataSet:
                dataSet[row[0]] = dict()
            #Place the value for the statistic for the appropriate team in the dictionary
            dataSet[row[0]][name] = row[1]

    return dataSet

"""
Breaks the compiled data into features.
Each feature is calculated by substracting the statistic for the second team listed from the first team listed
Each label is either a 1 or a 2 indicating that "Team One" won or "Team Two" won
"""
def breakData():
    csvFile = open('Data/train.csv', 'rU')
    reader = csv.reader(csvFile)

    features = []
    labels = []
    i = 0
    for row in reader:
        if (i == 0):
            i += 1
            continue
        labels.append(row[2])
        space = [float(row[3]) - float(row[9]), float(row[4]) - float(row[10]), float(row[5]) - float(row[11]), float(row[6]) - float(row[12]), float(row[7]) - float(row[13]), float(row[8]) - float(row[14])]
        features.append(space)
    return features, labels


from sklearn.model_selection import cross_val_score

#Cross validates the classifier to check accuracy
def check_classifier(clf, features, labels):
    #Computes scores using 10-fold crossvalidation
    scores = cross_val_score(clf, features, labels, cv=10)
    print scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
    print "\n"

#Loads the classifer from a pickle
def load_classifier(name="optimal.pkl"):
    return pickle.load(open(name))

#Helper function to save the classifier to a pickle file
def save_classifier(clf, features, labels, name="optimal.pkl"):
    clf.fit(features, labels)
    pickle.dump(clf, open(name, 'w+'))


def main():
    features, labels = breakData()
    #Train an SVM on with these parameters
    svc = svm.SVC(kernel='linear', C=1000, gamma=0.1)
    check_classifier(svc, features, labels)
    save_classifier(svc, features, labels)

if __name__ == "__main__":
    main()
