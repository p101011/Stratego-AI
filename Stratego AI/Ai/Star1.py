from player import *
from pieces import *


class MonteCarloTreeSearch(Player):
    def __init__(self, team, board):
        Player.__init__(self, team, board)

    def getMove(self):
        pass

    def star1(self, state, action, depth, alpha, beta):
        if depth == 0 or self.isTerminal(state):
            return self.boardEvaluator(state)

