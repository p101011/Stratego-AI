from player import *
from pieces import *
from random import choice, sample


class MonteCarloTreeSearch(Player):
    def __init__(self, team, board):
        Player.__init__(self, team, board)
        self.upper_bound = 2150  # derived from eval functions
        self.lower_bound = 0
        self.sample_resolution = 5

    def getMove(self):
        newBoard = self.relaxBoard(self.board)
        successors = self.generateAllMoves(newBoard, self.getTeam()[0])
        maximum = float('-inf')
        bestMove = None
        for start in successors:
            for end in successors[start]:
                succ_board = self.getNewStates(start, end, newBoard)
                value = self.expectiminimax(succ_board, 3, 'max', float('-inf'), float('inf'), False)
                if value > maximum:
                    maximum = value
                    bestMove = (start, end)
        return bestMove

    def expectiminimax(self, boards, depth, node_type, alpha, beta, consider_subset):
        if depth == 0:
            return self.boardEvaluator(boards[0][0])

        if len(boards) > 1:  # there is an uncertainty here
            best = 0
            for board, probability in boards:
                board_actions = self.generateAllMoves(board, self.getTeam()[0])
                if not consider_subset:
                    for start in board_actions:
                        for end in board_actions[start]:
                            best += probability * self.star2ss(board, (start, end), depth, alpha, beta, node_type)
                else:
                    action = (board_actions[0], board_actions[0][0])
                    best += probability * self.star2ss(board, action, depth, alpha, beta, node_type)
            return best
        elif self.isTerminal(boards[0][0]):
            return self.boardEvaluator(boards[0][0])
        elif node_type == 'max':
            return self.maxValue(boards[0][0], alpha, beta, depth)
        else:
            return self.minValue(boards[0][0], alpha, beta, depth)

    def star2ss(self, state, action, depth, alpha, beta, to_play):
        if self.isTerminal(state) or depth == 0:
            return self.boardEvaluator(state)

        outcomes = self.getNewStates(action[0], action[1], state)
        opti = alpha
        pess = beta

        for o, p in outcomes:
            ov = self.boardEvaluator(o) * p
            opti = max(opti, ov)
            pess = min(pess, ov)

        for outcome, probability in outcomes:
            alpha_mod = (alpha - opti + (probability * self.boardEvaluator(outcome))) / probability
            ap = max(self.lower_bound, alpha_mod)
            beta_mod = (beta - pess + (probability * self.boardEvaluator(outcome))) / probability
            bp = min(self.upper_bound, beta_mod)
            v = self.expectiminimax([(outcome, probability)], depth - 1, to_play, ap, bp, True)
            if to_play == 'max':
                pess = min(v * probability, pess)
                if pess >= beta:
                    return pess
            else:
                opti = max(v * probability, opti)
                if opti <= alpha:
                    return opti

        vsum = 0

        for outcome, probability in outcomes:
            alpha_mod = (alpha - opti + (probability * self.boardEvaluator(outcome))) / probability
            ap = max(self.lower_bound, alpha_mod)
            beta_mod = (beta - pess + (probability * self.boardEvaluator(outcome))) / probability
            bp = min(self.upper_bound, beta_mod)
            v = self.expectiminimax([(outcome, probability)], depth - 1, to_play, ap, bp, False)
            opti = max(v * probability, opti)
            pess = min(v * probability, pess)
            if v >= bp:
                return pess
            if v <= ap:
                return opti
            vsum += v

        if len(outcomes) == 0:
            return self.boardEvaluator(state)

        return vsum / len(outcomes)

    def maxValue(self, board, alpha, beta, depth):
        successors = self.generateAllMoves(board, self.getTeam()[0])
        bestVal = alpha
        br = False
        for start in successors:
            for end in successors[start]:
                newBoard = self.getNewState(start, end, board)
                childVal = self.star2ss(newBoard, (start, end), depth - 1, bestVal, beta, 'min')
                bestVal = max(childVal, bestVal)
                if beta <= bestVal:
                    br = True
                    break
            if br:
                break
        return bestVal

    def minValue(self, board, alpha, beta, depth):
        team = self.getTeam()[0]
        if team == 'r':
            otherTeam = 'b'
        else:
            otherTeam = 'r'
        br = False

        successors = self.generateAllMoves(board, otherTeam)
        bestVal = beta
        for start in successors:
            for end in successors[start]:
                newBoard = self.getNewState(start, end, board)
                childVal = self.star2ss(newBoard, (start, end), depth - 1, alpha, bestVal, 'max')
                bestVal = min(childVal, bestVal)
                if bestVal <= alpha:
                    br = True
                    break
            if br:
                break
        return bestVal

    def probe(self, state, alpha, beta):
        if self.isTerminal(state):
            return self.boardEvaluator(state)

        children = self.generateAllMoves(state, self.getTeam()[0])
        random_start = choice(children.keys())

        while len(children[random_start]) == 0:
            random_start = choice(children.keys())

        random_end = choice(children[random_start])
        board = self.getNewState(random_start, random_end, state)
        v = self.boardEvaluator(board)
        return min(v, alpha, beta)

    def relaxBoard(self, board):
        # replaces instances of pieces with a string ID, e.g. blue miner = '3b'
        newBoard = []
        for i in range(8):
            newBoard.append([])
            for j in range(8):
                piece = board[i][j]
                if not piece is None:
                    team = piece.getTeam()
                    rank = piece.getRank()
                    ID = str(rank) + team[0]
                    newBoard[i].append(ID)
                else:
                    newBoard[i].append(None)
        return newBoard

    def isTerminal(self, board):
        # returns True if the game with the given board is over (flag captured or no movable pieces)
        flags = 0
        movablePieces = 0
        for row in board:
            for piece in row:
                if piece is not None:
                    rank, team = self.getInfo(piece)
                    if rank == 0:
                        flags += 1
                    elif not rank == 11:
                        movablePieces += 1
                    else:
                        pass

        if flags < 2 or movablePieces == 0:
            return True
        else:
            return False

    def evaluateBoard1(self, board):
        otherFlag = False
        team = self.getTeam()[0]
        for i in range(8):
            for j in range(8):
                piece = board[i][j]
                if piece is not None:
                    pieceRank, pieceTeam = self.getInfo(piece)
                    if pieceRank == 0 and not pieceTeam == team:
                        otherFlag = True

        if otherFlag:
            return 0
        else:
            return 10

    def evaluateBoard2(self, board):
        myPieces = 0
        enemyPieces = 0
        myTeam = self.getTeam()[0]

        for i in range(8):
            for j in range(8):
                piece = board[i][j]
                if not piece is None:
                    pieceRank, pieceTeam = self.getInfo(piece)
                    if pieceTeam == myTeam:
                        myPieces += 1
                    else:
                        enemyPieces += 1

        return myPieces - enemyPieces

    def evaluateBoard3(self, board):
        advance = 0
        row = 0
        myTeam = self.getTeam()[0]
        if myTeam == 'b':
            weight = 0
        else:
            weight = 7

        for i in range(8):
            for piece in board[i]:
                if not piece is None:
                    rank, team = self.getInfo(piece)
                    if team == myTeam:
                        row += 1
            advance += abs(weight - i) * (row * 3)
            row = 0

        return advance

    def boardEvaluator(self, board):
        endgame = self.evaluateBoard1(board) * 10  # max is 100
        pieces = self.evaluateBoard2(board) * 1  # max is 10
        advance = self.evaluateBoard3(board) * 10  # max is 2040

        return endgame + pieces + advance

    def generateAllMoves(self, board, teamID):
        # generates all moves possible for this team
        # returns dict: {pieceLocation : list of possible moves for that piece}
        allMoves = {}

        for row in range(8):
            for col in range(8):
                piece = board[row][col]
                if not piece is None:
                    rank, pieceTeam = self.getInfo(piece)
                    if pieceTeam == teamID:
                        start = (row, col)
                        moves = self.getAllMoves(start, board)
                        allMoves[start] = moves
        return allMoves

    def getAllMoves(self, start, board):
        # gets a list of all possible end spots for this start spot
        row, col = start[0], start[1]
        myPiece = board[row][col]
        myRank, myTeam = self.getInfo(myPiece)
        moves = []

        if myRank == 0 or myRank == 11:
            return moves

        if myRank == 2:
            r, c = row, col
            while r > 0:
                r = r - 1
                possibleSpot = board[r][c]
                rank, team = self.getInfo(possibleSpot)
                if possibleSpot is None:  # if possibleSpot is open
                    moves.append((r, c))
                elif not team == myTeam:  # if possibleSpot is occupied by enemy
                    moves.append((r, c))
                    break
                else:
                    break

            r, c = row, col
            while c > 0:
                c = c - 1
                possibleSpot = board[r][c]
                rank, team = self.getInfo(possibleSpot)
                if possibleSpot is None:
                    moves.append((r, c))
                elif not team == myTeam:
                    moves.append((r, c))
                    break
                else:
                    break

            r, c = row, col
            while r < 7:
                r = r + 1
                possibleSpot = board[r][c]
                rank, team = self.getInfo(possibleSpot)
                if possibleSpot is None:
                    moves.append((r, c))
                elif not team == myTeam:
                    moves.append((r, c))
                    break
                else:
                    break

            r, c = row, col
            while c < 7:
                c = c + 1
                possibleSpot = board[r][c]
                rank, team = self.getInfo(possibleSpot)
                if possibleSpot is None:
                    moves.append((r, c))
                elif not team == myTeam:
                    moves.append((r, c))
                    break
                else:
                    break

            return moves
        else:
            r, c = row, col
            if r > 0:
                possibleSpot = board[r - 1][c]
                rank, team = self.getInfo(possibleSpot)
                if possibleSpot is None or not team == myTeam:
                    moves.append((r - 1, c))

            if c > 0:
                possibleSpot = board[r][c - 1]
                rank, team = self.getInfo(possibleSpot)
                if possibleSpot is None or not team == myTeam:
                    moves.append((r, c - 1))

            if r < 7:
                possibleSpot = board[r + 1][c]
                rank, team = self.getInfo(possibleSpot)
                if possibleSpot is None or not team == myTeam:
                    moves.append((r + 1, c))

            if c < 7:
                possibleSpot = board[r][c + 1]
                rank, team = self.getInfo(possibleSpot)
                if possibleSpot is None or not team == myTeam:
                    moves.append((r, c + 1))

        return moves

    def getNewState(self, start, end, board):
        # returns a new board after the given move was made
        newBoard = []
        for row in range(8):
            newBoard.append([])
            for col in range(8):
                newBoard[row].append(board[row][col])

        myPiece = newBoard[start[0]][start[1]]
        myRank, myTeam = self.getInfo(myPiece)

        enemyPiece = newBoard[end[0]][end[1]]
        if enemyPiece is not None:
            enemyRank, enemyTeam = self.getInfo(enemyPiece)

        newBoard[start[0]][start[1]] = None
        if enemyPiece is None:
            newBoard[end[0]][end[1]] = myPiece
        elif enemyRank < myRank:
            # win
            newBoard[end[0]][end[1]] = myPiece
        elif enemyRank == myRank:
            # tie
            newBoard[end[0]][end[1]] = None
        else:
            # lost
            pass

        return newBoard

    def getNewStates(self, start, end, board):
        # returns a new board after the given move was made
        newBoards = []

        enemyPiece = board[end[0]][end[1]]
        if enemyPiece is not None:
            possibilities = self.get_possibilities()
            for possibility, probability in possibilities:
                new_board = []

                for row in range(8):
                    new_board.append([])
                    for col in range(8):
                        new_board[row].append(board[row][col])

                myPiece = new_board[start[0]][start[1]]
                myRank, myTeam = self.getInfo(myPiece)

                new_board[start[0]][start[1]] = None
                if enemyPiece is None:
                    new_board[end[0]][end[1]] = myPiece
                elif possibility < myRank:
                    # win
                    new_board[end[0]][end[1]] = myPiece
                elif possibility == myRank:
                    # tie
                    new_board[end[0]][end[1]] = None
                else:
                    # lost
                    pass

                newBoards.append((new_board, probability))
        else:
            new_board = []
            probability = 1.0

            for row in range(8):
                new_board.append([])
                for col in range(8):
                    new_board[row].append(board[row][col])

            myPiece = new_board[start[0]][start[1]]
            new_board[start[0]][start[1]] = None
            new_board[end[0]][end[1]] = myPiece
            newBoards.append((new_board, probability))
        return newBoards

    def getInfo(self, piece):
        # returns the rank of the piece and its teamID
        # format of piece is '9b', '11r', etc.
        if piece is None:
            rank = None
            teamID = None
        elif len(piece) == 3:
            rank = int(piece[0] + piece[1])
            teamID = piece[2]
        else:
            rank = int(piece[0])
            teamID = piece[1]
        return rank, teamID

    def get_possibilities(self):
        output = []
        dummy = ProbabilityDistribution()
        for rank in sample(dummy.ranks.keys(), self.sample_resolution):
            output.append((dummy.ranks[rank], 1.0 / self.sample_resolution))
        return output
