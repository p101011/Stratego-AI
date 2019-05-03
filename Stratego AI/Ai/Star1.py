from player import *
from pieces import *


class Star1(Player):
    def __init__(self, team, board):
        Player.__init__(self, team, board)
        self.upper_bound = 2150  # derived from eval functions
        self.lower_bound = 0

    def getMove(self):
        newBoard = self.relaxBoard(self.board)
        successors = self.generateAllMoves(newBoard, self.getTeam()[0])
        maximum = float('-inf')
        bestMove = None
        for start in successors:
            for end in successors[start]:
                succ_board = self.getNewStates(start, end, newBoard)
                value = self.expectiminimax(succ_board, 3, 'min', float('-inf'), float('inf'))
                if value > maximum:
                    maximum = value
                    bestMove = (start, end)
        return bestMove

    def expectiminimax(self, boards, depth, node_type, alpha, beta):
        if depth == 0:
            return self.boardEvaluator(boards[0][0])
        if len(boards) > 1:  # there is an uncertainty here
            best = 0
            for board, probability in boards:
                best += probability * self.star1(board, depth, alpha, beta, node_type)
            return best
        elif self.isTerminal(boards[0][0]):
            return self.boardEvaluator(boards[0][0])
        elif node_type == 'max':
            return self.maxValue(boards[0][0], alpha, beta, depth)
        else:
            return self.minValue(boards[0][0], alpha, beta, depth)

    def star1(self, state, depth, alpha, beta, node_type):
        if depth <= 0 or self.isTerminal(state):
            return self.boardEvaluator(state)

        successors = self.generateAllMoves(state, self.getTeam()[0])
        A = len(successors) * (alpha - self.upper_bound) + self.upper_bound
        B = len(successors) * (beta - self.lower_bound) + self.lower_bound
        vsum = 0
        for start in successors:
            for end in successors[start]:
                options = self.getNewStates(start, end, state)
                v = sum([self.boardEvaluator(x[0]) for x in options]) / len(options)
                ax = max(A, self.lower_bound)
                bx = min(B, self.upper_bound)
                if node_type == 'max':
                    v = max(v, ax, bx)
                else:
                    v = min(v, ax, bx)
                if not v > A:
                    return A
                if not v < B:
                    return B
                vsum += v
                A = A + self.upper_bound - v
                B = B + self.lower_bound - v
        return vsum / len(successors)

    def maxValue(self, board, alpha, beta, depth):
        successors = self.generateAllMoves(board, self.getTeam()[0])
        bestVal = alpha
        br = False
        for start in successors:
            for end in successors[start]:
                newBoard = self.getNewStates(start, end, board)
                childVal = self.expectiminimax(newBoard, depth - 1, 'min', bestVal, beta)
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
                newBoard = self.getNewStates(start, end, board)
                childVal = self.expectiminimax(newBoard, depth - 1, 'max', alpha, bestVal)
                bestVal = min(childVal, bestVal)
                if bestVal <= alpha:
                    br = True
                    break
            if br:
                break
        return bestVal

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

    def getNewStates(self, start, end, board):
        # returns a new board after the given move was made
        newBoards = []

        enemyPiece = board[end[0]][end[1]]
        if enemyPiece is not None:
            possibilities = get_possibilities()
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


def get_possibilities():
    output = []
    dummy = ProbabilityDistribution()
    for rank in dummy.ranks.keys():
        probability = dummy.distribution[rank] / float(dummy.activePieces) > 0
        if probability > 0:
            output.append((dummy.ranks[rank], probability))
    return output
