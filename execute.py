#!/usr/bin/env python -W ignore::DeprecationWarning

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import sklearn
    import data_extract as de

def load(year):
    # Load points per game (ppg), offensive efficiency (oe), defensive efficiency (de), 
    # field goal efficiency (fge), offensive rebounds (or), and rpi
    knowledge = de.constructKnowledge(['ppg', 'oe', 'de', 'fge', 'or', 'rpi'], year)
    clf = de.load_classifier()
    return knowledge, clf

#Returns the corresponding statistics for team queried
def query(team, knowledge):
    if team not in knowledge:
        return False

    teamData = knowledge[team]
    return [teamData['ppg'], teamData['oe'], teamData['de'], teamData['fge'], teamData['or'], teamData['rpi']]

#Calculates the features of the selected game by subtracting the statistics of the two teams
def gameData(one_data, two_data):
    data = []
    for row in range(0, len(one_data)):
        data.append(float(one_data[row]) - float(two_data[row]))
    return data

def prediction(team_one, team_two, knowledge, clf):
    dataOne = query(team_one, knowledge)
    dataTwo = query(team_two, knowledge)
    if not dataOne:
        return "Could not find %s in knowledgebase" % team_one
    if not dataTwo:
        return "Could not find %s in knowledgebase" % team_two

    #Calculate the feature
    feature_space = gameData(dataOne, dataTwo)
    #Use the classifier to predict the winner. 1 for team one and 2 for team two
    pred = clf.predict(feature_space)
    if pred[0] == '1':
        return team_one
    else:
        return team_two

#Create a bracket by entering each matchup in the tournament
def question(clf, knowledge):
    team_one = raw_input("Enter Team One: ")
    team_two = raw_input("Enter Team Two: ")

    message = prediction(team_one, team_two, knowledge, clf)
    print message
    shouldContinue = raw_input("Do you want to continue (Y/N): ")
    if shouldContinue == "Y":
        question(clf, knowledge)
    else:
        return

#Load the knowledge and the classifier
knowledge, clf = load("2017-2018")
question(clf, knowledge)
