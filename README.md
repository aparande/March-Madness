# March Madness Bracket Maker

## Background
The NCAA March Madness tournament is notorious for having game outcomes which are almost impossible to predict because underdog teams beat the favorites.
However, there are statistics about each team in the tournament which can be used to predict the winner of a game and thus create a bracket.

## Data Files
### 2018 and 2019 Tournaments
All data is contained in the `MasterData.xlsx` spreadsheet.

The `GameData` sheet contains the matchups from the 2016, 2017, and 2018 tournaments.

The sheets named with a specific season (i.e 2018-2019) contain the statistics for each team in the NCAA for that season

All data was taken from https://www.teamrankings.com/ncb/team-stats/

### Tournaments 2021 onwards
All data was taken from [Kaggle](https://www.kaggle.com/) using the dataset from that year.

To download the data via the Kaggle CLI
```sh
kaggle competitions download march-machine-learning-mania-2024 -p data
unzip -d data/kaggle_data march-machine-learning-mania-2024.zip
```

Once the data is downloaded, you can run the feature extraction via
```sh
python3 summarize_kaggle.py
```

## Getting started for a new year
1. Set up the directory for the year
```
mkdir <year>
cp <prev-year>/Training.ipynb <year>
```
2. Train a classifier via the new `Training.ipynb` we just created and save the
   model. Only SK learn classifiers are currently supported currently.
```
python3 main.py create --year <year> --predictor sklearn_seed --predictor-path <path-to-pkl>
```
3. Inspect the bracket
```
python3 main.py visualize <bracket-path>
```

## Results
- [2018 Results](2018/README.md)
- [2019 Results](2019/README.md)
- [2021 Results](2021/README.md)
- [2022 Results](2022/README.md)
- [2023 Results](2023/README.md)
