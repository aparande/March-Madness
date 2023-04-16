from bracket_tree import build_tree
from classifier_utils import load_classifier
from kaggle_data_utils import build_team_lookup

import numpy as np

def predict_lda(lda_file, use_seed, data_means=None, data_stds=None):
  clf = load_classifier(lda_file)
  lookup = build_team_lookup()

  def pred(winning_team, losing_team):
    winner, win_seed = winning_team
    loser, lose_seed = losing_team

    data_win = lookup.get(winner, None)
    data_lose = lookup.get(loser, None)

    if data_win is None or data_lose is None:
      raise ValueError(f"Either {winner} or {loser} not found")

    features = data_win - data_lose
    features = np.hstack(([int(win_seed) - int(lose_seed)], features))

    if data_means is not None:
      features -= data_means

    if data_stds is not None:
      features /= data_stds

    if not use_seed:
      features = features[1:]
    features = features.reshape((1, -1))

    prob_loser_wins, prob_winner_wins = clf.predict_proba(features)[0]

    return prob_winner_wins

  return pred

def determine_winners(bracket, out_file="out.txt"):
  queue = [bracket]
  level_count = 0
  next_level = 1
  level_games = []
  last_team = None
  with open(out_file, "w") as f:
    while len(queue) != 0:
      node = queue.pop(0)

      optimal_team = max(node.teams, key=lambda x: node.dist[x])[0]
      level_count += 1

      if last_team is not None:
        level_games.append((last_team, optimal_team))
        last_team = None
      else:
        last_team = optimal_team

      if level_count == next_level:
        if last_team is not None:
          level_games.append((last_team))
        f.writelines([str(level_games) + "\n"])
        next_level *= 2
        level_count = 0
        level_games = []
        last_team = None

      if node.left_node is not None and node.right_node is not None:
        queue.append(node.left_node)
        queue.append(node.right_node)

if __name__ == '__main__':
  means = np.load("2021/data-mean.npy")
  stds = np.load("2021/data-std.npy")
  predict_func = predict_lda("2021/lda.pkl", True, data_means=means, data_stds=stds)

  bracket = build_tree("2021/bracket.txt")
  bracket.compute_distribution(predict_func)
  determine_winners(bracket, out_file="2021/lda-output.txt")
