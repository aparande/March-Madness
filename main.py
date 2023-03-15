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


def _create_sklearn_featurizer(year: int, seed_lookup: dict[str, bracket_types.Seed]) -> predictors.SkLearnFeaturizer:
    lookup = kaggle_data_utils.build_team_lookup(year)
    means = np.load(f"{year}/data-mean.npy")
    stds = np.load(f"{year}/data-std.npy")

    return predictors.SkLearnFeaturizer(lookup, seed_lookup, means, stds)

def _create_high_seed_bracket(args: argparse.Namespace) -> networkx.DiGraph:
    seed_lookup = kaggle_data_utils.get_seeds(args.year)
    predictor = predictors.HighSeedPredictor(seed_lookup)
    bracket_builder = builder.BracketBuilder(predictor)
    return bracket_builder.build(seed_lookup)

def _create_sklearn_seed_bracket(args: argparse.Namespace) -> networkx.DiGraph:
    seed_lookup = kaggle_data_utils.get_seeds(args.year)
    featurizer = _create_sklearn_featurizer(args.year, seed_lookup)
    predictor = predictors.SkLearnSeedPredictor(args.predictor_path, featurizer)
    bracket_builder = builder.BracketBuilder(predictor)
    return bracket_builder.build(seed_lookup)

def _create_bracket_args(subparser):
    parser = subparser.add_parser('create', help='Create a bracket')
    parser.add_argument('name', type=str)
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--predictor', type=predictors.Predictors, choices=list(predictors.Predictors), required=True)
    parser.add_argument('--predictor-path', type=str,
                        help='Path to the predictor model. Required depending on the predictor type')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--out-path', type=pathlib.Path, default=None)

def _create_and_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(help='Options', required=True, dest='cmd')
    _create_bracket_args(subparsers)
    
    return parser.parse_args()

def create_bracket(args: argparse.Namespace):
    bracket: networkx.DiGraph | None = None
    if args.predictor == predictors.Predictors.SKLEARN_SEED:
        assert args.predictor_path, f'--predictor-path is required for {args.predictor}'
        bracket = _create_sklearn_seed_bracket(args)
    elif args.predictor == predictors.Predictors.HIGH_SEED:
        bracket = _create_high_seed_bracket(args)

    if not bracket:
        return

    if args.visualize:
        labels: dict[str, str] = {}
        for node in bracket:
            game = bracket.nodes[node][bracket_types.GAME_DATA_KEY]
            labels[node] = f'{game.winner}'

        layout = networkx.multipartite_layout(bracket, subset_key='round_num', align='vertical', scale=4)
        networkx.draw_networkx(bracket, labels=labels, node_size=50, font_size=8, pos=layout, verticalalignment='bottom', font_weight='bold')
        plt.show()

    if args.out_path is not None:
        builder.BracketBuilder.write_to_csv(bracket, args.name, args.out_path)


def main():
    args = _create_and_parse_args()

    if args.cmd == 'create':
        create_bracket(args)

    # prob_func = predictors.SkLearnProbabilityFunction(args.prob_model_path, featurizer)
    # evaluator = evaluate.SinglePerturbationRobustnessEvaluator(prob_func, predictor)
    # score = evaluator.evaluate(bracket)
    # print(f'Bracket score is {score}')

if __name__ == '__main__':
    main()
