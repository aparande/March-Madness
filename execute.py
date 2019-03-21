#!/usr/bin/env python -W ignore::DeprecationWarning
from builder import BracketBuilder

if __name__ == "__main__":
    builder = BracketBuilder("2018-2019", ["adaboost_clf.pkl", "svm.pkl","svm-noseed.pkl"], [True, True, False])
    builder.build_interactive()
