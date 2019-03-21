from data_extract import *
import numpy as np
class BracketBuilder:
    def __init__(self, year, clf_files, use_seeds):
        self.knowledge = construct_knowledge([year])[year]
        self.clfs = []
        for filename in clf_files:
            self.clfs.append(load_classifier(name=filename))
        self.use_seeds = use_seeds

    def build_interactive(self):
        team_one = input("Enter Team One: ")
        team_two = input("Enter Team Two: ")
        seed_one = None
        seed_two = None
        if any(self.use_seeds):
            seed_one = input("Enter Team One's Seed: ")
            seed_two = input("Enter Team Two's Seed: ")

        message = self.prediction(team_one, team_two, seed_one, seed_two)
        print(message)
        shouldContinue = input("Do you want to continue (Y/N): ")
        if shouldContinue == "Y":
            self.build_interactive()
        else:
            return

    def prediction(self, team_one, team_two, seed_one, seed_two):
        dataOne = self.query(team_one)
        dataTwo = self.query(team_two)
        if dataOne is None:
            return "Could not find %s in knowledgebase" % team_one
        if dataTwo is None:
            return "Could not find %s in knowledgebase" % team_two

        #Calculate the feature
        feature_space = self.gameData(dataOne, dataTwo, seed_one, seed_two)
        feature_space = np.reshape(feature_space, (1, -1))
        #Use the classifier to predict the winner. 1 for team one and 2 for team two
        votes = []

        for clf, use_seed in zip(self.clfs, self.use_seeds):
            features = feature_space
            if not use_seed:
                features = features[:, :-1]
            votes.append(clf.predict(features)[0])

        avgVote = sum(votes) / len(votes)
        print(votes)
        if avgVote > 1.5:
            return team_two
        else:
            return team_one

    def gameData(self, one_data, two_data, seed_one, seed_two):
        data = one_data - two_data
        
        if seed_one and seed_two:
            if seed_one < seed_two:
                data = np.append(data, 1)
            elif seed_two < seed_one:
                data = np.append(data, 2)
            else:
                data = np.append(data, random.randint(1, 2))

        return data

    def query(self, team):
        team_stats = self.knowledge.loc[self.knowledge["Team Name"] == team].iloc[:, 1:]
        if len(team_stats.index) == 0:
            return None

        return team_stats.values