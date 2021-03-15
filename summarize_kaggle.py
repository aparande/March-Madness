import pandas as pd

# Script to process Kaggle Data

GAME_STATS = ["FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA", "OR", "DR", "Ast", "TO", "Stl", "Blk", "PF", "Score", "TeamID"] 
WINNER_STATS = [f"W{stat}" for stat in GAME_STATS]
LOSER_STATS = [f"L{stat}" for stat in GAME_STATS]

def load_teams():
  teams_df = pd.read_csv("data/kaggle_data/MTeams.csv")
  return teams_df[["TeamID", "TeamName"]]

def load_seeds():
  seeds_df = pd.read_csv("data/kaggle_data/MNCAATourneySeeds.csv")
  seeds_df = seeds_df[seeds_df.Season >= 2003]
  seeds_df = seeds_df[~seeds_df.Seed.str.contains("a") & ~seeds_df.Seed.str.contains("b")]
  seeds_df["Seed"] = seeds_df["Seed"].apply(lambda x: int(x[1:]))
  return seeds_df

def load_tourney_games_compact():
  games_df = pd.read_csv("data/kaggle_data/MNCAATourneyCompactResults.csv")
  games_df = games_df[["Season", "WTeamID", "LTeamID"]]
  return games_df[games_df.Season >= 2003]

def summarize_tournaments():
  seeds = load_seeds()
  games = load_tourney_games_compact()
  win_seeds = pd.merge(games, seeds, how='inner', left_on=["Season", "WTeamID"], right_on=["Season", "TeamID"])
  out = pd.merge(win_seeds, seeds, how='inner', left_on=["Season", "LTeamID"], right_on=["Season", "TeamID"], suffixes=["_W", "_L"])

  out = out[["Season", "WTeamID", "LTeamID", "Seed_W", "Seed_L"]]
  return out

def load_regular_games():
  games_df = pd.read_csv("data/kaggle_data/MRegularSeasonDetailedResults.csv")
  return games_df.drop(labels=["DayNum", "WLoc", "NumOT"], axis=1)

def process_raw_stats(df, prefix):
  """
  Select the columns pertaining to prefix (either W or L) and do some basic processing
  """
  # Remove unnecessary columns
  stats_df = df.drop(labels = LOSER_STATS if prefix == "W" else WINNER_STATS, axis=1)
  stats_df[GAME_STATS] = stats_df[WINNER_STATS if prefix == "W" else LOSER_STATS]
  stats_df = stats_df.drop(labels = WINNER_STATS if prefix == "W" else LOSER_STATS, axis=1)

  # Compute 2 Pointers Made and Attempted
  stats_df["FGM2"] = stats_df["FGM"] - stats_df["FGM3"]
  stats_df["FGA2"] = stats_df["FGA"] - stats_df["FGA3"]

  # Compute Posessions
  stats_df["Pos"] = stats_df["FGA"] - stats_df["OR"] + stats_df["TO"] + 0.45 * stats_df["FTA"]

  stats_df["Opp_DR"] = df["LDR" if prefix == "W" else "WDR"]
  stats_df["Opp_Score"] = df["LScore" if prefix == "W" else "WScore"]
  stats_df["Win"] = 1 if prefix == "W" else 0

  return stats_df
  
def compute_stats():
  all_games = load_regular_games()

  winner_df = process_raw_stats(all_games, "W")
  loser_df = process_raw_stats(all_games, "L")

  stats_df = pd.concat([winner_df, loser_df], ignore_index=True)
  sum_stats = stats_df.groupby(["Season", "TeamID"], as_index=False).sum()
  sum_stats = sum_stats.drop(labels=["Win"], axis=1)

  stats_df["Margin"] = stats_df["Score"] - stats_df["Opp_Score"]
  means = stats_df.groupby(["Season", "TeamID"], as_index=False).mean()[["Season", "TeamID", "Win", "Score", "Margin"]]

  all_stats = pd.merge(sum_stats, means, on=["Season", "TeamID"], how='inner', suffixes=["_sum", "_mean"])
 
  all_stats["PPG"] = all_stats["Score_mean"]
  all_stats["WPP"] = all_stats["Win"] # Win Percentage
  all_stats["OE"] = all_stats["Score_sum"] / all_stats["Pos"] # Offensive Efficiency
  all_stats["DE"] = all_stats["Opp_Score"] / all_stats["Pos"] # Defensive Efficiency
  all_stats["FGE"] = (all_stats["FGM2"] + 1.5 * all_stats["FGM3"]) / all_stats["FGA"] # Field Goal Efficiency
  all_stats["OR_per"] = all_stats["OR"] / (all_stats["OR"] + all_stats["Opp_DR"]) # Offensive Rebounds
  all_stats["EPR"] = (all_stats["Pos"] + all_stats["OR"] - all_stats["TO"]) / all_stats["Pos"] # Effective Possession Ratio

  all_stats.drop(labels=["Win"], axis=1)

  return all_stats[["Season", "TeamID", "PPG", "WPP", "OE", "DE", "FGE", "OR_per", "EPR", "Margin"]]

if __name__ == '__main__':
  tournaments = summarize_tournaments()
  tournaments.to_csv("data/processed-data/tourney-games.csv", index=False)

  stats = compute_stats()
  stats.to_csv("data/processed-data/all-season-stats.csv", index=False)

