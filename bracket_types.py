"""
Module for Bracket construction
"""

import abc
from dataclasses import dataclass
from typing import NamedTuple

GAME_DATA_KEY = 'game_data'

class Seed(NamedTuple):
    region: str
    num: int
    play_in: str | None

    def __str__(self):
        return f'{self.region}{self.num}{self.play_in or ""}'

    @classmethod
    def from_kaggle_str(cls, seed_str: str) -> 'Seed':
        play_in = None
        if len(seed_str) == 4:
            play_in = seed_str[3]
        region = seed_str[0]
        num = int(seed_str[1:3])
        return cls(region=region, num=num, play_in=play_in)

class PredictionFunction(abc.ABC):
    @abc.abstractmethod
    def predict(self, team_one: str, team_two: str) -> str:
        pass

    def __call__(self, team_one: str, team_two: str) -> str:
        return self.predict(team_one, team_two)

class ProbabilityFunction(abc.ABC):
    @abc.abstractmethod
    def get_prob(self, team_one: str, team_two: str) -> dict[str, float]:
        pass

    def __call__(self, team_one: str, team_two: str) -> dict[str, float]:
        return self.get_prob(team_one, team_two)

@dataclass
class GameData:
    team_one: str
    team_two: str
    round_num: int
    winner: str | None = None

    def swap_winner(self) -> None:
        if self.winner == self.team_one:
            self.winner = self.team_two
        else:
            self.winner = self.team_one

