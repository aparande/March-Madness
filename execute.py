#!/usr/bin/env python -W ignore::DeprecationWarning
from builder import BracketBuilder
from kaggle_data_utils import build_team_lookup
import numpy as np

if __name__=='__main__':
  lookup = build_team_lookup()
  means = np.load("2021/data-mean.npy")
  stds = np.load("2021/data-std.npy")
  builder = BracketBuilder(lookup, ["2021/svm.pkl"], [True], data_means=means, data_stds=stds)
  builder.build_interactive()
