"""
Module for bracket evaluation
"""

import abc
from copy import deepcopy

import networkx

import bracket_types


class BracketEvaluator(abc.ABC):
    """Abstract class for evaluating brackets represented as a networkx graph"""

    @abc.abstractmethod
    def evaluate(self, bracket: networkx.DiGraph, verbose: bool = False) -> float:
        """Evaluate a bracket

        Args:
            bracket: A network X DiGraph representing a bracket. Each node in
                the graph is a game should have an associated GameData object

        Returns:
            A float representing the score of the bracket.
        """
        pass

    def __call__(self, bracket: networkx.DiGraph, verbose: bool = False) -> float:
        return self.evaluate(bracket, verbose)


class SinglePerturbationRobustnessEvaluator(BracketEvaluator):
    """Bracket evaluate for "Single Perturbation Robustness

    The idea behind single perturbation robustness is a single incorrect
    prediction should have a limited impact on the correctness of the rest of
    the bracket. 

    For each game in the bracket
    1. Suppose the bracket is wrong, and generate the resulting bracket using the prediction function. 
    2. Compute the score of the new bracket and compute the error incurred assuming the original bracket is 100% correct
    3. Weight the error by the probability of it occurring based on a probability model
    4. Sum together all the weighted single perturbation errors
    """
    def __init__(self, probability_func: bracket_types.ProbabilityFunction, prediction_func: bracket_types.PredictionFunction) -> None:
        self.probability_func = probability_func
        self.prediction_func = prediction_func

    def _apply_perturbation(self, game_name, subbracket: networkx.DiGraph):
        next_game_node = game_name
        perturbation: tuple[str, str] = "", ""
        # breakpoint()
        while next_game_node is not None:
            current_game_node = next_game_node
            game_data: bracket_types.GameData = subbracket.nodes[current_game_node][bracket_types.GAME_DATA_KEY]
            assert game_data.winner
            if next_game_node == game_name:
                # For the perturbation, swap the winner
                old_winner = game_data.winner
                game_data.swap_winner()
                perturbation = old_winner, game_data.winner
            else:
                old_winner = game_data.winner

                # Apply the previous perturbation
                if perturbation[0] == game_data.team_one:
                    game_data.team_one = perturbation[1]
                else:
                    game_data.team_two = perturbation[1]

                game_data.winner = self.prediction_func(game_data.team_one, game_data.team_two)
                if old_winner == game_data.winner:
                    # The perturbation no longer has an impact, so end the loop
                    break

                perturbation = old_winner, game_data.winner

            next_game_node = None
            for next_node in subbracket.successors(current_game_node):
                assert next_game_node is None, "Current game node has multiple successors, which is impossible"
                next_game_node = next_node

    def _evaluate_perturbation(self, node, original: networkx.DiGraph, perturbed: networkx.DiGraph, verbose: bool) -> int:
        next_game_node = node
        error: int = 0
        perturbation_str = ""
        while next_game_node is not None:
            current_game_node, next_game_node = next_game_node, None

            perturbed_game: bracket_types.GameData = perturbed.nodes[current_game_node][bracket_types.GAME_DATA_KEY]
            original_game: bracket_types.GameData = original.nodes[current_game_node][bracket_types.GAME_DATA_KEY]
            if perturbed_game.winner != original_game.winner:
                perturbation_str += f'-> R{perturbed_game.round_num} {perturbed_game.winner} '
                if perturbed_game.round_num != 0:
                    error += 10 * (2 ** (perturbed_game.round_num - 1))

            for next_node in perturbed.successors(current_game_node):
                assert next_game_node is None, "Current game node has multiple successors, which is impossible"
                next_game_node = next_node

        if verbose:
            print(perturbation_str)
        return error

    def evaluate(self, bracket: networkx.DiGraph, verbose: bool = False) -> float:
        score: float = 0
        for node in bracket:
            descendents = networkx.descendants(bracket, node)
            descendents.add(node)
            original_subbracket: networkx.DiGraph = bracket.subgraph(descendents)
            perturbed_subbracket: networkx.DiGraph = deepcopy(original_subbracket)

            self._apply_perturbation(node, perturbed_subbracket)
            error = self._evaluate_perturbation(node, original_subbracket, perturbed_subbracket, verbose)

            game_data: bracket_types.GameData = original_subbracket.nodes[node][bracket_types.GAME_DATA_KEY]
            assert game_data.winner
            if verbose:
                print(game_data)
            prob = self.probability_func(game_data.team_one, game_data.team_two)
            perturbed_prob = 1 - prob[game_data.winner]
            if verbose:
                print(f'Perturbation Probability: {perturbed_prob}')

            score += perturbed_prob * error

        return score



