class TreeNode:
  def __init__(self, left_node = None, right_node = None):
    if left_node is not None and right_node is not None:
      self.teams = left_node.teams.union(right_node.teams)

    self.left_node = left_node
    self.right_node = right_node
    self.dist = None

  def compute_distribution(self, predict_func):
    if self.dist is not None:
      return self.dist

    self.dist = dict()
    left_dist = self.left_node.compute_distribution(predict_func)
    right_dist = self.right_node.compute_distribution(predict_func)

    for team in self.teams:
      prob = 0.0
      if team in left_dist:
        for other in right_dist:
          prob += predict_func(team, other) * left_dist[team] * right_dist[other]
      elif team in right_dist:
        for other in left_dist:
          prob += predict_func(team, other) * right_dist[team] * left_dist[other]

      self.dist[team] = prob

    return self.dist



class LeafNode(TreeNode):
  def __init__(self, team):
    self.teams = set([team])
    super().__init__()

  def compute_distribution(self, predict_func):
    self.dist = {team: 1 for team in self.teams}
    return self.dist

def build_tree(team_file):
  with open(team_file, "r") as f:
    lines = f.readlines()

  nodes = [LeafNode(tuple(line.split(","))) for line in lines]
  while len(nodes) > 1:
    left = nodes.pop(0)
    right = nodes.pop(0)
    nodes.append(TreeNode(left_node=left, right_node=right))

  return nodes[0]

