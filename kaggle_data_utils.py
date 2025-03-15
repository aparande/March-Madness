from __future__ import annotations

import dataclasses
import json
import pathlib
from collections import defaultdict
from typing import cast

import numpy as np
import pandas as pd
import sklearn.utils

import bracket_types

RANDOM_STATE = 42
# Features used prior to 2024
PRE2024_FEATURES = ["Seed", "PPG", "WPP", "OE", "DE", "FGE", "OR_per", "EPR", "Margin"]


@dataclasses.dataclass
class DatasetDescription:
    feature_names: list[str]
    normalized: bool
    means_path: str | None
    std_path: str | None

    def __post_init__(self):
        if self.normalized:
            assert self.means_path
            assert self.std_path

    @classmethod
    def from_file(cls, file_path: pathlib.Path) -> DatasetDescription:
        return DatasetDescription(**json.loads(file_path.read_text()))

class Dataset():
    _trainX: np.ndarray
    _trainY: np.ndarray

    def __init__(self, features: list[str], relative_path: str='.',
                 one_hot_labels: bool = False, normalize: bool = True):
        self._feature_names = features[:]
        self._normalized = normalize

        self._build_data(relative_path, one_hot_labels)

    @property
    def feature_names(self):
        return self._feature_names[:]

    @property
    def trainX(self) -> np.ndarray:
        return self._trainX

    @property
    def trainY(self) -> np.ndarray:
        return self._trainY

    @property
    def means(self) -> np.ndarray:
        if self._normalized:
            return self._means
        else:
            return self._trainX.mean(axis=0)

    @property
    def std(self) -> np.ndarray:
        if self._normalized:
            return self._std
        else:
            return self._trainX.std(axis=0)

    def _build_data(self, relative_path: str, one_hot_labels: bool):
        season_stats = pd.read_csv(f"{relative_path}/data/processed-data/all-season-stats.csv")
        tournament_games = pd.read_csv(f"{relative_path}/data/processed-data/tourney-games.csv")

        df = pd.merge(tournament_games, season_stats, how="inner",
                      left_on=["Season", "WTeamID"], right_on=["Season",
                                                               "TeamID"])
        df = pd.merge(df, season_stats, how="inner", left_on=["Season",
                                                              "LTeamID"],
                      right_on=["Season", "TeamID"], suffixes=["_W", "_L"])

        for feature in self._feature_names:
            df[feature] = df[f"{feature}_W"] - df[f"{feature}_L"]

        trainX = df[self.feature_names].values

        # Generate labels of -1 and 1
        trainY: np.ndarray = 2 * np.random.randint(2, size=trainX.shape[0]) - 1 
        # Randomly decide whether team 1 wins or team 2 wins using the label
        # (otherwise team 1 always wins)
        trainX: np.ndarray = trainX * trainY[:, np.newaxis] 

        if one_hot_labels:
            trainY = (trainY + 1) // 2

        self._trainX, self._trainY = sklearn.utils.shuffle(trainX, trainY, random_state=RANDOM_STATE) # type: ignore

        if self._normalized:
            self._means = cast(np.ndarray, self._trainX).mean(axis=0)
            self._std = cast(np.ndarray, self._trainX).std(axis=0)

            self._trainX = (self._trainX - self._means) / self._std

    def save_description(self, out_dir: str, name: str):
        mean_path: str | None = None
        std_path: str | None = None
        if self._normalized:
            mean_path = f'{out_dir}/{name}-data-mean.npy'
            np.save(mean_path, self.means)
            std_path = f'{out_dir}/{name}-data-std.npy'
            np.save(std_path, self.std)

        desc = DatasetDescription(self.feature_names, self._normalized, mean_path, std_path)

        with open(f'{out_dir}/{name}.json', 'w') as f:
            json.dump(dataclasses.asdict(desc), f)

def build_dataset(relative_path=".", out_path="../2022", normalize = False, one_hot_labels = False):
  """(LEGACY) Builds a dataset using the Kaggle Data

  relative_path is the path to the data directory
  normalize controls whether or not the features are normalized
  one_hot_labels determines whether the labels are one-hot encoded. If one_hot_encoded, 0 = team 2 wins and 1 = team 1 wins, otherwie -1 = team 2 wins and 1 = team 1 wins.
  """

  dataset = Dataset(PRE2024_FEATURES, relative_path=relative_path,
                    one_hot_labels=one_hot_labels, normalize=normalize)
  if normalize:
      dataset.save_description(out_path, '')

  return dataset.trainX, dataset.trainY

def build_team_lookup(year: int, feature_names: list[str]) -> dict[str, np.ndarray]:
  teams = pd.read_csv("data/kaggle_data/MTeams.csv").drop(labels=["FirstD1Season", "LastD1Season"], axis=1)
  stats = pd.read_csv("data/processed-data/all-season-stats.csv")
  stats = stats[stats.Season == year]

  lookup = pd.merge(teams, stats, on="TeamID", how="inner")
  lookup_dict = dict()

  # TODO: should build Seed into the team lookup. Then everything downstream
  # can be seed agnostic.
  feature_names.remove('Seed')
  for row, data in lookup.iterrows():
    lookup_dict[data["TeamName"]] = data.loc[feature_names].values

  return lookup_dict

def get_seeds(year: int, relative_path='.') -> dict[str, bracket_types.Seed]:
  teams = pd.read_csv(f'{relative_path}/data/kaggle_data/MTeams.csv').drop(labels=['FirstD1Season', 'LastD1Season'], axis=1)
  seeds = pd.read_csv(f'{relative_path}/data/kaggle_data/MNCAATourneySeeds.csv')
  seeds = seeds[seeds.Season == year]

  lookup = pd.merge(teams, seeds, on="TeamID", how="inner")

  lookup_dict: dict[str, bracket_types.Seed] = dict()
  for row, data in lookup.iterrows():
    lookup_dict[cast(str, data["TeamName"])] = bracket_types.Seed.from_kaggle_str(cast(str, data["Seed"]))
    
  return lookup_dict

def build_seed_win_probabilities(relative_path='.') -> dict[int, dict[int, float]]:
    seeds = pd.read_csv(f'{relative_path}/data/kaggle_data/MNCAATourneySeeds.csv')
    seeds['Seed'] = seeds['Seed'].transform(lambda x: int(x[1:3]))

    games_df = pd.read_csv(f"{relative_path}/data/kaggle_data/MNCAATourneyCompactResults.csv")
    games_df = games_df[["Season", "WTeamID", "LTeamID"]]

    games_with_seeds = games_df.merge(seeds, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='inner')
    games_with_seeds['WSeed'] = games_with_seeds['Seed']
    games_with_seeds = games_with_seeds.drop(labels=['Seed', 'TeamID'], axis=1)
    games_with_seeds = games_with_seeds.merge(seeds, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='inner')
    games_with_seeds['LSeed'] = games_with_seeds['Seed']
    games_with_seeds = games_with_seeds.drop(labels=['Seed', 'TeamID'], axis=1)

    probs = defaultdict(dict)
    for i in range(1, 16):
        for j in range(i + 1, 17):
            # Compute how many times seed i has beaten seed j
            wins_df = games_with_seeds[(games_with_seeds.WSeed == i) & (games_with_seeds.LSeed == j)]
            lose_df = games_with_seeds[(games_with_seeds.LSeed == i) & (games_with_seeds.WSeed == j)]

            win_count = len(wins_df)
            lose_count = len(lose_df)
            game_count = win_count + lose_count

            if game_count == 0:
                continue

            probs[i][j] = win_count / game_count
            probs[j][i] = lose_count / game_count

    return probs
