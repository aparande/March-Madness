# Methodology
The methodology has remained largely the same as in previous years. Using the
Kaggle dataset, I compute important statistics for each team using their
regular season data.

- Offensive Efficiency (oe)
- Defensive Efficiency (de)
- Field Goal Efficiency (fge)
- Offensive Rebounds (or)
- Effective Possession Ratio (epr)
- Win Percentage (wp)
The seed of each team was also used as a feature to the algorithm.

The actual features that were fed into the algorithms as training data were the difference of the statistics between the two teams.
Each algorithm would predict if the first team or the second team would win the match.

## Metrics
See [the 2022 metrics section](../2022/README.md)

## Classifiers
I trained an SVM on all of the features, including seed.

The baseline model to compare against was a High Seed model which always picks
the higher seed to win and does a coin flip if both seeds are equal

# Results
![2024 Raw Data](../imgs/2024-data.png)

![2024 Cumulative Scores](../imgs/2024-scores.png)

