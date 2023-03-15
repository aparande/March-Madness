#!/usr/bin/env python -W ignore::DeprecationWarning
import sys

import numpy as np
import torch

from builder import BracketBuilder
from classifier_utils import load_classifier
from kaggle_data_utils import build_team_lookup


def predict_sklearn(clf_files, use_seeds):
  clfs = [load_classifier(name=filename) for filename in clf_files]

  def predict_func(features, team_one, team_two):
    #Use the classifier to predict the winner. 1 for team one and 2 for team two
    votes = []

    for clf, use_seed in zip(clfs, use_seeds):
        if not use_seed:
          features = features[:, 1:]
        votes.append(clf.predict(features))

    avgVote = np.mean(votes)
    print(votes)
    if avgVote > 0:
        return team_one
    else:
        return team_two

  return predict_func

def predict_seeded_fallback(model_name):
  seeded = load_classifier(f"{model_name}.pkl")
  unseeded = load_classifier(f"{model_name}-noseed.pkl")

  def predict_func(features, team_one, team_two):
    #Use the classifier to predict the winner. 1 for team one and 2 for team two
    decision = unseeded.predict(features[:, 1:])
    if decision != -1 * unseeded.predict(-1 * features[:, 1:]):
      print("Falling Back on Seeded Model!")
      decision = seeded.predict(features)

    if decision > 0:
      return team_one
    else:
      return team_two

  return predict_func


def predict_torch(clf_file, use_seed):
  clf = load_classifier(name=clf_file) 
  clf.eval()

  def predict_func(features, team_one, team_two):
    #Use the classifier to predict the winner. 1 for team one and 2 for team two
    if not use_seed:
      features = features[:, 1:]

    features = torch.from_numpy(features.astype(np.float32))

    pred = clf(features).argmax(1).item()

    if pred == 1:
        return team_one
    else:
        return team_two

  return predict_func

if __name__=='__main__':
  year = int(sys.argv[1])
  model = sys.argv[2]
  lookup = build_team_lookup(year)
  means = np.load(f"{year}/data-mean.npy")
  stds = np.load(f"{year}/data-std.npy")

  # predict_func = predict_sklearn([f"{year}/{model}.pkl"], [False])
  # predict_func = predict_torch("2021/nn-noseed.mdl", False)
  predict_func = predict_seeded_fallback(f"{year}/{model}")
  builder = BracketBuilder(lookup, predict_func, data_means=means, data_stds=stds)
  builder.build_interactive()
