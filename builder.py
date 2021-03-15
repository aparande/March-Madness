from classifier_utils import load_classifier
import numpy as np

class BracketBuilder:
    def __init__(self, season_data, clf_files, use_seeds, data_means=None, data_stds=None):
        """
        season_data is a dictionary mapping team name to team stats
        """
        self.knowledge = season_data
        self.data_means = data_means
        self.data_stds = data_stds

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
        dataOne = self.knowledge.get(team_one, None)
        dataTwo = self.knowledge.get(team_two, None)

        if dataOne is None:
            return f"Could not find {team_one} in knowledgebase" 
        if dataTwo is None:
            return f"Could not find {team_two} in knowledgebase"

        #Calculate the feature
        feature_space = self.gameData(dataOne, dataTwo, seed_one, seed_two)
        #Use the classifier to predict the winner. 1 for team one and 2 for team two
        votes = []

        for clf, use_seed in zip(self.clfs, self.use_seeds):
            features = feature_space
            if not use_seed:
              features = features[:, 1:]
            votes.append(clf.predict(features))

        avgVote = np.mean(votes)
        print(votes)
        if avgVote > 0:
            return team_one
        else:
            return team_two

    def gameData(self, one_data, two_data, seed_one, seed_two):
        data = one_data - two_data
        data = np.hstack([[int(seed_one) - int(seed_two)], data])
        if self.data_means is not None:
          data = data - self.data_means
        if self.data_stds is not None:
          data = data / self.data_stds
        return data.reshape((1, -1)) # SKLearn Requirement

