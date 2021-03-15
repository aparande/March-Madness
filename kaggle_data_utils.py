import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import sklearn.utils

FEATURES = ["Seed", "PPG", "WPP", "OE", "DE", "FGE", "OR_per", "EPR", "Margin"]

def build_dataset(relative_path=".", normalize=False):
  season_stats = pd.read_csv(f"{relative_path}/data/processed-data/all-season-stats.csv")
  tournament_games = pd.read_csv(f"{relative_path}/data/processed-data/tourney-games.csv")

  df = pd.merge(tournament_games, season_stats, how="inner", left_on=["Season", "WTeamID"], right_on=["Season", "TeamID"])
  df = pd.merge(df, season_stats, how="inner", left_on=["Season", "LTeamID"], right_on=["Season", "TeamID"], suffixes=["_W", "_L"])

  for feature in FEATURES:
    df[feature] = df[f"{feature}_W"] - df[f"{feature}_L"]

  trainX = df[FEATURES].values

  trainY = 2 * np.random.randint(2, size=trainX.shape[0]) - 1 # Generate labels of -1 and 1
  trainX = trainX * trainY[:, np.newaxis] # Randomly decide whether team 1 wins or team 2 wins using the label (otherwise team 1 always wins)

  trainX, trainY = sklearn.utils.shuffle(trainX, trainY)

  if normalize:
    trainX = (trainX - trainX.mean(axis=0)) / trainX.std(axis=0)
    np.save(f"{relative_path}/2021/data-mean.npy", trainX.mean(axis=0))
    np.save(f"{relative_path}/2021/data-std.npy", trainX.std(axis=0))
  return trainX, trainY

def build_team_lookup():
  teams = pd.read_csv("data/kaggle_data/MTeams.csv").drop(labels=["FirstD1Season", "LastD1Season"], axis=1)
  stats = pd.read_csv("data/processed-data/all-season-stats.csv")
  stats = stats[stats.Season == 2020]

  lookup = pd.merge(teams, stats, on="TeamID", how="inner")
  lookup_dict = dict()
  for row, data in lookup.iterrows():
    lookup_dict[data["TeamName"]] = data.iloc[3:].values

  return lookup_dict




