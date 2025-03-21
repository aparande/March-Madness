import pickle
import sys

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

# TODO: Remove workaround for my strange Python environment
if sys.version_info.minor <= 9:
    import torch

#Cross validates the classifier to check accuracy
def check_classifier(clf, features, labels):
    #Computes scores using 10-fold crossvalidation
    print("Cross-Validating model")
    scores = cross_val_score(clf, features, labels, cv=10)
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
    print("\n")

#Loads the classifer from a pickle
def load_classifier(name="optimal.pkl"):
  if name.split(".")[-1] == "pkl":
    with open(name, "rb") as f:
      return pickle.load(f)
  else:
    return torch.load(name)

#Helper function to save the classifier to a pickle file
def save_classifier(clf, features, labels, name="optimal.pkl"):
    clf.fit(features, labels)
    pickle.dump(clf, open(name, 'wb+'))

def grid_search_params(clf, features, labels, params):
    print("Searching Param Space")
    grid_cv = GridSearchCV(clf, param_grid=params, cv=5)
    grid_cv.fit(features, labels)
    print(grid_cv.best_params_)
