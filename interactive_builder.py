#!/usr/bin/env python -W ignore::DeprecationWarning
from builder import BracketBuilder
from classifier_utils import load_classifier

from kaggle_data_utils import build_team_lookup
import numpy as np
import torch

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
  lookup = build_team_lookup()
  means = np.load("2021/data-mean.npy")
  stds = np.load("2021/data-std.npy")

  # predict_func = predict_sklearn(["2021/svm-noseed.pkl"], [False])
  predict_func = predict_torch("2021/nn-noseed.mdl", False)
  builder = BracketBuilder(lookup, predict_func, data_means=means, data_stds=stds)
  builder.build_interactive()
