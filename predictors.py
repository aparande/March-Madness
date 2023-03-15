import enum
import random

import numpy as np

import bracket_types
import classifier_utils
import kaggle_data_utils


class Predictors(enum.Enum):
    SKLEARN_SEED = "sklearn_seed"
    HIGH_SEED = "high_seed"

    def __str__(self) -> str:
        return self.value

class ProbabilityModels(enum.Enum):
    SKLEARN = "sklearn"
    HIGH_SEED = "high_seed"

    def __str__(self) -> str:
        return self.value

class HighSeedPredictor(bracket_types.PredictionFunction):
    def __init__(self, seed_lookup: dict[str, bracket_types.Seed]) -> None:
        self.seed_lookup = seed_lookup

    def predict(self, team_one: str, team_two: str) -> str:
        seed_one = self.seed_lookup[team_one]
        seed_two = self.seed_lookup[team_two]

        if seed_one.num < seed_two.num:
            return team_one
        elif seed_two.num < seed_one.num:
            return team_two
        else:
            t1_wins = random.random() > 0.5
            return team_one if t1_wins else team_two

class SkLearnFeaturizer:
    def __init__(self, team_lookup: dict[str, np.ndarray], seed_lookup: dict[str, bracket_types.Seed], data_means: np.ndarray | None = None, data_stds: np.ndarray | None = None) -> None:
        self.team_lookup = team_lookup
        self.seed_lookup = seed_lookup
        self.data_means = data_means
        self.data_stds = data_stds

    def _make_features(self, team_one: str, team_two: str) -> np.ndarray:
        data_one = self.team_lookup[team_one]
        data_two = self.team_lookup[team_two]
        seed_one = self.seed_lookup[team_one]
        seed_two = self.seed_lookup[team_two]
        

        features = data_one - data_two
        features = np.hstack([[int(seed_one.num) - int(seed_two.num)], features])
        if self.data_means is not None:
          features = features - self.data_means
        if self.data_stds is not None:
          features = features / self.data_stds
        return features.reshape((1, -1)) # SKLearn Requirement

    def __call__(self, team_one: str, team_two: str):
        return self._make_features(team_one, team_two)

class SkLearnSeedPredictor(bracket_types.PredictionFunction):
    def __init__(self, clf_path: str, featurizer: SkLearnFeaturizer) -> None:
        self.clf = classifier_utils.load_classifier(clf_path)
        self.featurizer = featurizer

    def predict(self, team_one: str, team_two: str) -> str:
        features = self.featurizer(team_one, team_two)
        if self.clf.predict(features)> 0:
            return team_one
        else:
            return team_two

class SkLearnProbabilityFunction(bracket_types.ProbabilityFunction):
    def __init__(self, clf_path: str, featurizer: SkLearnFeaturizer) -> None:
        self.clf = classifier_utils.load_classifier(clf_path)
        self.featurizer = featurizer

    def get_prob(self, team_one: str, team_two: str) -> dict[str, float]:
        features = self.featurizer(team_one, team_two)
        prob = self.clf.predict_proba(features)[0]
        out: dict[str, float] = {}
        for label, p in zip(self.clf.classes_, prob):
            team = team_one if label == 1 else team_two
            out[team] = p
        return out

class HighSeedProbabilityFunction(bracket_types.ProbabilityFunction):
    def __init__(self, seed_lookup: dict[str, bracket_types.Seed]) -> None:
        self.seed_lookup = seed_lookup
        self.probability_table = kaggle_data_utils.build_seed_win_probabilities()

    def get_prob(self, team_one: str, team_two: str) -> dict[str, float]:
        seed_one = self.seed_lookup[team_one]
        seed_two = self.seed_lookup[team_two]

        if seed_one.num in self.probability_table and seed_two.num in self.probability_table[seed_one.num]:
            return {
                team_one: self.probability_table[seed_one.num][seed_two.num],
                team_two: 1 - self.probability_table[seed_one.num][seed_two.num]
            }
        else:
            return {team_one: 0.5, team_two: 0.5}

