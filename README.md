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

### 2021 Tournament
All data was taken from [Kaggle](https://www.kaggle.com/c/ncaam-march-mania-2021).

## Python Files
- `custom_data_utils.py` contains helper methods for extracting data, training the classifier, and saving it to disk for custom data stored in the `MasterData.xlsx` spreadsheet.
- `builder.py` contains a class called BracketBuilder which encapsulates the functionality of building a bracket based on SKLearn or PyTorch classifiers.
- `interactive_builder.py` creates a BracketBuilder object and begins prompting the user to build a bracket.
- `dist_builder.py` build a bracket using an LDA classifier to predict probabilities. It then runs Maximimum Likelihood Estimation to generate the winners of each match.
- `kaggle_data_utils.py` contains helper methods for processing data from the Kaggle March Madness dataset.

## Results
- [2018 Results](2018/README.md)
- [2019 Results](2019/README.md)
- [2021 Results](2021/README.md)
- [2022 Results](2022/README.md)
