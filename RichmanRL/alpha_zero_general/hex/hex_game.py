import numpy as np
from RichmanRL.envs import HexBoard

class HexMCTSRandomGame():

    def __init__(self):
        pass
    
    def getInitBoard(self):
        return np.zeros((11,11))
    
    def getBoardSize(self):
        return (11,11)
    
    def getActionSize(self):
        return 121
    
    def getNextState(self, board, action1, action2):
        roll = 1 if np.random.random() < 0.5 else 2
        player, action = (1, action1) if roll == 1 else (2, action2)
        ret = np.copy(board)
        if(ret[action//11, action%11] != 0):
            raise ValueError("Illegal move made in HEX MCTS")
        ret[action//11, action%11] = player
        return ret, player
    
    def getCanonicalForm(self, board):
        b1 = np.zeros((11,11))
        mask = board == 1
        b1[mask] = 1
        b2 = np.zeros((11,11))
        mask = board == 2
        b2[mask] = 1
        ret1 = np.concatenate((b1[:, :, np.newaxis], b2[:, :, np.newaxis]), axis=2)
        ret2 = np.concatenate((b2[:, :, np.newaxis], b1[:, :, np.newaxis]), axis=2)
        return ret1, ret2

    def getSymmetries(self, canonicalBoard, pi):
        return [(canonicalBoard, pi)]
    
    def getGameEnded(self, board):
        hexWrapper = HexBoard(11)
        for i in range(11):
            for j in range(11):
                if(board[i,j] == 0):
                    continue
                hexWrapper.play_action(i, j, board[i, j])
        if(not hexWrapper.check_game_over()):
            return 0
        return 1 if hexWrapper.win(1) else 2
    
    def stringRepresentation(self, canonicalBoard):
        board = canonicalBoard[:,:,0] + 2*canonicalBoard[:,:,1]
        return "".join([str(int(i)) for i in board.flatten()])
    
    def getValidMoves(self, board):
        ret = np.ones((11,11))
        mask = board != 0
        ret[mask] = 0
        return ret.flatten()