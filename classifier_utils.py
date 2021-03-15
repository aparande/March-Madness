import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report

import pickle

#Cross validates the classifier to check accuracy
def check_classifier(clf, features, labels):
    #Computes scores using 10-fold crossvalidation
    print("Cross-Validating model")
    scores = cross_val_score(clf, features, labels, cv=10)
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
    print("\n")

#Loads the classifer from a pickle
def load_classifier(name="optimal.pkl"):
    with open(name, "rb") as f:
        return pickle.load(f)

#Helper function to save the classifier to a pickle file
def save_classifier(clf, features, labels, name="optimal.pkl"):
    clf.fit(features, labels)
    pickle.dump(clf, open(name, 'wb+'))

def grid_search_params(clf, features, labels, params):
    print("Searching Param Space")
    grid_cv = GridSearchCV(clf, param_grid=params, cv=5)
    grid_cv.fit(features, labels)
    print(grid_cv.best_params_)
