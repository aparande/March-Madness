#!/usr/bin/env python -W ignore::DeprecationWarning

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import sklearn
    import data_extract as de
import random
import numpy as np

def load(year, model_name):
    # Load points per game (ppg), offensive efficiency (oe), defensive efficiency (de), 
    # field goal efficiency (fge), offensive rebounds (or), and rpi
    knowledge = de.construct_knowledge([year])[year]
    clf = de.load_classifier(model_name)
    return knowledge, clf

#Returns the corresponding statistics for team queried
def query(team, knowledge):
    team_stats = knowledge.loc[knowledge["Team Name"] == team].iloc[:, 1:]
    if len(team_stats.index) == 0:
        return None

    return team_stats.values

#Calculates the features of the selected game by subtracting the statistics of the two teams
def gameData(one_data, two_data, seed_one, seed_two):
    data = one_data - two_data
    
    if seed_one < seed_two:
        data = np.append(data, 1)
    elif seed_two < seed_one:
        data = np.append(data, 2)
    else:
        data = np.append(data, random.randint(1, 2))

    return data

def prediction(team_one, team_two, seed_one, seed_two, knowledge, clf):
    dataOne = query(team_one, knowledge)
    dataTwo = query(team_two, knowledge)
    if dataOne is None:
        return "Could not find %s in knowledgebase" % team_one
    if dataTwo is None:
        return "Could not find %s in knowledgebase" % team_two

    #Calculate the feature
    feature_space = gameData(dataOne, dataTwo, seed_one, seed_two)
    feature_space = np.reshape(feature_space, (1, -1))
    #Use the classifier to predict the winner. 1 for team one and 2 for team two
    pred = clf.predict(feature_space)
    if pred[0] == 1:
        return team_one
    else:
        return team_two

#Create a bracket by entering each matchup in the tournament
def question(clf, knowledge):
    team_one = input("Enter Team One: ")
    team_two = input("Enter Team Two: ")
    seed_one = input("Enter Team One's Seed: ")
    seed_two = input("Enter Team Two's Seed: ")

    message = prediction(team_one, team_two, seed_one, seed_two, knowledge, clf)
    print(message)
    shouldContinue = input("Do you want to continue (Y/N): ")
    if shouldContinue == "Y":
        question(clf, knowledge)
    else:
        return

#Load the knowledge and the classifier
knowledge, clf = load("2018-2019", "adaboost_clf.pkl")
question(clf, knowledge)
