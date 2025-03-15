import argparse
import pathlib

import matplotlib.pyplot as plt
import networkx
import numpy as np

import bracket_types
import builder
import evaluate
import kaggle_data_utils
import predictors


def _visualize(bracket:  networkx.DiGraph):
    labels: dict[str, str] = {}
    for node in bracket:
        print(node, bracket.nodes[node])
        game: bracket_types.GameData = bracket.nodes[node][bracket_types.GAME_DATA_KEY]
        labels[node] = f'{game.winner}'

    layout = networkx.multipartite_layout(bracket, subset_key='round_num', align='vertical', scale=4)
    networkx.draw_networkx(bracket, labels=labels, node_size=50, font_size=8, pos=layout, verticalalignment='bottom', font_weight='bold')
    plt.show()

def _create_sklearn_seed_featurizer(year: int, seed_lookup: dict[str, bracket_types.Seed], 
                               dataset_descriptor: pathlib.Path) -> predictors.SkLearnSeedFeaturizer:
    dataset_description = kaggle_data_utils.DatasetDescription.from_file(dataset_descriptor)
    lookup = kaggle_data_utils.build_team_lookup(year, dataset_description.feature_names)

    means = None
    if dataset_description.means_path:
        means = np.load(dataset_description.means_path)

    stds = None
    if dataset_description.std_path:
        stds = np.load(dataset_description.std_path)

    return predictors.SkLearnSeedFeaturizer(lookup, seed_lookup, means, stds)

def _create_sklearn_featurizer(year: int, dataset_descriptor: pathlib.Path) -> predictors.SkLearnFeaturizer:
    dataset_description = kaggle_data_utils.DatasetDescription.from_file(dataset_descriptor)
    lookup = kaggle_data_utils.build_team_lookup(year, dataset_description.feature_names)

    means = None
    if dataset_description.means_path:
        means = np.load(dataset_description.means_path)

    stds = None
    if dataset_description.std_path:
        stds = np.load(dataset_description.std_path)

    return predictors.SkLearnFeaturizer(lookup, means, stds)

def _create_high_seed_bracket_predictor(args: argparse.Namespace, seed_lookup: dict[str, bracket_types.Seed]) -> predictors.HighSeedPredictor:
    seed_lookup = kaggle_data_utils.get_seeds(args.year)
    return predictors.HighSeedPredictor(seed_lookup)

def _create_sklearn_seed_bracket_predictor(year: int, predictor_path: pathlib.Path,
                                           dataset_descriptor: pathlib.Path,
                                           seed_lookup: dict[str, bracket_types.Seed]) -> predictors.SkLearnPredictor:
    seed_lookup = kaggle_data_utils.get_seeds(year)
    featurizer = _create_sklearn_seed_featurizer(year, seed_lookup, dataset_descriptor)
    # TODO: Switch to using paths
    return predictors.SkLearnPredictor(str(predictor_path), featurizer)

def _create_sklearn_bracket_predictor(year: int, predictor_path: pathlib.Path,
                                           dataset_descriptor: pathlib.Path) -> predictors.SkLearnPredictor:
    featurizer = _create_sklearn_featurizer(year, dataset_descriptor)
    # TODO: Switch to using paths
    return predictors.SkLearnPredictor(str(predictor_path), featurizer)

def _create_bracket_args(subparser):
    parser = subparser.add_parser('create', help='Create a bracket')
    parser.add_argument('name', type=str)
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--predictor', type=predictors.Predictors, choices=list(predictors.Predictors), required=True)
    parser.add_argument('--predictor-path', type=pathlib.Path,
                        help='Path to the predictor model. Required depending on the predictor type')
    parser.add_argument('--dataset-descriptor', type=pathlib.Path,
                        help='Path to the dataset description used to train the model. Required depending on the predictor type')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--out-path', type=pathlib.Path, default=None)

def _create_visualize_args(subparser):
    parser = subparser.add_parser('visualize', help='Visualize a bracket')
    parser.add_argument('bracket_path', type=pathlib.Path)

def _create_compare_args(subparser):
    parser = subparser.add_parser('compare', help='Compare brackets against a metric')
    parser.add_argument('bracket_paths', type=pathlib.Path, nargs='+')
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--truth-model', type=predictors.Predictors, choices=list(predictors.Predictors), required=True)
    parser.add_argument('--truth-model-path', type=str,
                        help='Path to the truth model. Required depending on the predictor type')
    parser.add_argument('--truth-dataset-descriptor', type=str,
                        help='Path to the truth model dataset descriptor. Required depending on the predictor type')
    parser.add_argument('--probability-model', type=predictors.ProbabilityModels, choices=list(predictors.ProbabilityModels), required=True)
    parser.add_argument('--probability-model-path', type=str,
                        help='Path to the probability model. Required depending on the probability model type')
    parser.add_argument('--probability-dataset-descriptor', type=str,
                        help='Path to the probability model dataset descriptor. Required depending on the predictor type')
    parser.add_argument('--verbose', action='store_true')

def _create_and_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(help='Options', required=True, dest='cmd')
    _create_bracket_args(subparsers)
    _create_visualize_args(subparsers)
    _create_compare_args(subparsers)
    
    return parser.parse_args()

def create_bracket(args: argparse.Namespace):
    predictor: bracket_types.PredictionFunction

    seed_lookup = kaggle_data_utils.get_seeds(args.year)

    if args.predictor == predictors.Predictors.SKLEARN_SEED:
        assert args.predictor_path, f'--predictor-path is required for {args.predictor}'
        assert args.dataset_descriptor, f'--dataset-descriptor is required for {args.predictor}'
        predictor = _create_sklearn_seed_bracket_predictor(args.year, args.predictor_path, 
                                                           args.dataset_descriptor, seed_lookup)
    elif args.predictor == predictors.Predictors.SKLEARN:
        assert args.predictor_path, f'--predictor-path is required for {args.predictor}'
        assert args.dataset_descriptor, f'--dataset-descriptor is required for {args.predictor}'
        predictor = _create_sklearn_bracket_predictor(args.year, args.predictor_path, 
                                                           args.dataset_descriptor)
    elif args.predictor == predictors.Predictors.HIGH_SEED:
        predictor = _create_high_seed_bracket_predictor(args, seed_lookup)
    else:
        predictor = _create_high_seed_bracket_predictor(args, seed_lookup)

    bracket_builder = builder.BracketBuilder(predictor)
    bracket = bracket_builder.build(seed_lookup)

    if args.visualize:
        _visualize(bracket)

    if args.out_path is not None:
        builder.BracketBuilder.write_to_csv(bracket, args.name, args.out_path)

def visualize_bracket(args: argparse.Namespace) -> None:
    bracket = builder.BracketBuilder.read_from_csv(args.bracket_path)
    _visualize(bracket)

def compare_brackets(args: argparse.Namespace) -> None:
    predictor: bracket_types.PredictionFunction
    prob_func: bracket_types.ProbabilityFunction

    seed_lookup = kaggle_data_utils.get_seeds(args.year)

    if args.truth_model == predictors.Predictors.SKLEARN_SEED:
        assert args.truth_model_path, f'--truth-model-path is required for {args.truth_model}'
        assert args.truth_dataset_descriptor, f'--truth-dataset-descriptor is required for {args.truth_model}'
        predictor = _create_sklearn_seed_bracket_predictor(args.year, args.truth_model_path, args.truth_dataset_descriptor, seed_lookup)
    elif args.truth_model == predictors.Predictors.HIGH_SEED:
        predictor = _create_high_seed_bracket_predictor(args, seed_lookup)
    else:
        predictor = _create_high_seed_bracket_predictor(args, seed_lookup)

    if args.probability_model == predictors.ProbabilityModels.SKLEARN:
        assert args.probability_model_path, f'--probability-model-path is required for {args.probability_model}'
        assert args.probability_dataset_descriptor, f'--probability-dataset-descriptor is required for {args.truth_model}'
        featurizer = _create_sklearn_seed_featurizer(args.year, seed_lookup, args.probability_dataset_descriptor)
        prob_func = predictors.SkLearnProbabilityFunction(args.probability_model_path, featurizer)
    elif args.probability_model == predictors.ProbabilityModels.HIGH_SEED:
        prob_func = predictors.HighSeedProbabilityFunction(seed_lookup)
    else:
        prob_func = predictors.HighSeedProbabilityFunction(seed_lookup)

    evaluator = evaluate.SinglePerturbationRobustnessEvaluator(prob_func, predictor)
    scores: dict[str, float] = {}
    for bracket_path in args.bracket_paths:
        bracket = builder.BracketBuilder.read_from_csv(bracket_path)
        scores[str(bracket_path)] = evaluator.evaluate(bracket, verbose=args.verbose)

    print(scores)

def main():
    args = _create_and_parse_args()

    if args.cmd == 'create':
        create_bracket(args)
    elif args.cmd == 'visualize':
        visualize_bracket(args)
    elif args.cmd == 'compare':
        compare_brackets(args)

if __name__ == '__main__':
    main()
