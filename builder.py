import csv
import pathlib
from collections import defaultdict

import networkx

import bracket_types

class BracketBuilder:
    CSV_FIELDS = ['game_name', 'team_one', 'team_two', 'winner']
    REGIONS = ['W', 'X', 'Y', 'Z']

    def __init__(self, predict_func: bracket_types.PredictionFunction):
        """
        season_data is a dictionary mapping team name to team stats
        """
        self.predict_func = predict_func

    def _build_first_round(self, graph: networkx.DiGraph, seeds: dict[str, bracket_types.Seed]) -> list[str]:
        regions: dict[str, list[str]] = defaultdict(lambda: ["" for _ in range(16)])
        play_in_winners: set[str] = set()

        for team_name, seed in seeds.items():
            region = seed.region

            if (opponent_name := regions[region][seed.num - 1]) and opponent_name != "":
                # Do the play-in games
                assert seed.play_in and seeds[opponent_name].play_in is not None
                winner = self.predict_func(team_name, opponent_name)
                game_data = bracket_types.GameData(team_name, opponent_name, round_num=0, winner=winner)
                graph.add_node(f'play_in_{region}', game_data=game_data, round_num=game_data.round_num)
                regions[region][seed.num - 1] = winner
                play_in_winners.add(winner)
            else:
                regions[region][seed.num - 1] = team_name

        first_round: list[str] = []
        for j, region in enumerate(self.REGIONS):
            region_list = regions[region]
            game_num = 8 * j
            for j in [0, 7, 4, 3, 5, 2, 6, 1]:
                team_one, team_two = region_list[j], region_list[-j-1]
                game_data = bracket_types.GameData(team_one, team_two, round_num=1)
                game_name = f'1-{game_num}'
                graph.add_node(game_name, game_data=game_data, round_num = game_data.round_num)
                if team_one in play_in_winners or team_two in play_in_winners:
                    graph.add_edge(f'play_in_{region}', game_name)

                first_round.append(game_name)
                game_num += 1

        return first_round


    def build(self, seeds: dict[str, bracket_types.Seed]) -> networkx.DiGraph:
        graph = networkx.DiGraph()
        nodes = self._build_first_round(graph, seeds)

        while len(nodes) != 1:
            game_one = nodes.pop(0)
            game_two = nodes.pop(0)

            game_one_data: bracket_types.GameData = graph.nodes[game_one][bracket_types.GAME_DATA_KEY]
            game_two_data: bracket_types.GameData = graph.nodes[game_two][bracket_types.GAME_DATA_KEY]

            game_one_data.winner = self.predict_func(game_one_data.team_one, game_one_data.team_two)
            game_two_data.winner = self.predict_func(game_two_data.team_one, game_two_data.team_two)

            assert game_one_data.round_num == game_two_data.round_num
            game_one_num = int(game_one.split('-')[1])
            game_two_num = int(game_two.split('-')[1])
            assert (game_one_num + 1) == game_two_num

            new_game_data = bracket_types.GameData(game_one_data.winner, game_two_data.winner, round_num=game_one_data.round_num + 1)
            new_game_name = f'{game_one_data.round_num + 1}-{game_one_num // 2}'
            graph.add_node(new_game_name, game_data=new_game_data, round_num=new_game_data.round_num)
            graph.add_edge(game_one, new_game_name)
            graph.add_edge(game_two, new_game_name)
            nodes.append(new_game_name)

        final = nodes.pop(0)
        final_data: bracket_types.GameData = graph.nodes[final][bracket_types.GAME_DATA_KEY]
        final_data.winner = self.predict_func(final_data.team_one, final_data.team_two)

        return graph

    @classmethod
    def write_to_csv(cls, bracket: networkx.DiGraph, name: str, outfile: pathlib.Path):
        rows: list[dict[str, str | None]] = []
        for node in bracket:
            game_data: bracket_types.GameData = bracket.nodes[node][bracket_types.GAME_DATA_KEY]
            row = {
                    'game_name': node,
                    'team_one': game_data.team_one,
                    'team_two': game_data.team_two,
                    }

            if game_data.winner is not None:
                row['winner'] = game_data.winner
            rows.append(row)

        def sort_func(x: dict[str, str | None]) -> tuple[int, int]:
            game_name = x['game_name']
            assert game_name
            comps = game_name.split('-')
            if 'play' in game_name:
                return 0, 0
            round_num, game_num = comps
            return int(round_num), int(game_num)

        sorted_rows = sorted(rows, key=sort_func)

        with open(outfile / f'{name}.csv', 'w') as f:
            writer = csv.DictWriter(f, cls.CSV_FIELDS)
            writer.writeheader()

            writer.writerows(sorted_rows)

    @classmethod
    def read_from_csv(cls, infile: pathlib.Path) -> networkx.DiGraph:
        bracket = networkx.DiGraph()
        with open(infile, 'r') as f:
            reader = csv.DictReader(f, cls.CSV_FIELDS)

            # Add all the nodes into the graph
            for i, game in enumerate(reader):
                if i == 0:
                    continue
                game_name = game['game_name']
                round_num: int = 0
                if 'play' not in game_name:
                    comps = game_name.split('-')
                    round_num = int(comps[0])

                game_data = bracket_types.GameData(team_one=game['team_one'], team_two=game['team_two'], round_num=round_num, winner=game.get('winner'))
                bracket.add_node(game_name, game_data=game_data, round_num=game_data.round_num)

        # Populate the play-in round edges
        for i, region in enumerate(cls.REGIONS):
            play_in_winner = bracket.nodes[f'play_in_{region}'][bracket_types.GAME_DATA_KEY].winner
            assert play_in_winner
            for j in range(8):
                game_num = 8 * i + j
                game_name = f'1-{game_num}'
                assert game_name in bracket

                game_data: bracket_types.GameData = bracket.nodes[game_name][bracket_types.GAME_DATA_KEY]
                if play_in_winner == game_data.team_one or play_in_winner == game_data.team_two:
                    bracket.add_edge(f'play_in_{region}', game_name)

        # Populate the rest of the round edges
        for round_num in range(1, 6):
            num_games = 2 ** (6 - round_num)
            for game_num in range(num_games):
                next_round_game = f'{round_num + 1}-{game_num // 2}'
                current_round_game = f'{round_num}-{game_num}'
                bracket.add_edge(current_round_game, next_round_game)

        return bracket


