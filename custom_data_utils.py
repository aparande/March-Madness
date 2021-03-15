import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn import svm

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import pickle

def construct_knowledge(years):
    """
    Compiles a dictionary of dataframes from the Excel spreadsheet
    """
    team_data = dict()
    for year in years:
        team_data[year] = read_from_sheet(year)
    return team_data
    

def read_from_sheet(sheet_name, header=0):
    """
    Reads data from a specified sheet in the Excel to a Pandas dataframe
    """
    df = pd.read_excel("MasterData.xlsx",sheet_name=sheet_name, header=header)
    return df

"""
Breaks the compiled data into features.
Each feature is calculated by substracting the statistic for the second team listed from the first team listed
Each label is either a 1 or a 2 indicating that "Team One" won or "Team Two" won
"""
def breakData():
    game_df = read_from_sheet("GameData")
    knowledge = construct_knowledge(["2017-2018", "2016-2017", "2015-2016"])

    game_num = len(game_df.index)
    features = np.zeros((game_num, 9))
    labels = np.zeros((game_num,))
    feature_num = 0
    for team_one, team_two, year, winner, high_seed in game_df[["Team One", "Team Two", "Year", "Winner", "HighSeed"]].values:
        year_knowledge = knowledge[year]
        one_df = year_knowledge.loc[year_knowledge["Team Name"] == team_one]
        if (len(one_df.index) == 0):
            print("Could not find data for %s:%s" % (team_one, year))
            continue
        
        two_df = year_knowledge.loc[year_knowledge["Team Name"] == team_two]
        if (len(two_df.index) == 0):
            print("Could not find data for %s:%s" % (team_two, year))
            continue

        features[feature_num, 0:8] = one_df.iloc[:, 1:] - two_df.iloc[:, 1:].values
        features[feature_num,8] = high_seed
        labels[feature_num] = winner
        feature_num += 1

    #Return the features and labels
    features = features[:feature_num]
    labels = labels[:feature_num]
    return features, labels

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
