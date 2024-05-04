import logging
import math
import time
import numpy as np

EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

    def getActionProb(self, board, player, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.numMCTSSims):
            self.search(board, player)
        s = self.game.stringRepresentation(self.game.getCanonicalForm(board)[player-1])
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]
        for a in range(121):
            if(s, a) in self.Nsa:
                assert s[a] == '0'


        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        if(counts_sum == 0):
            return self.game.getValidMoves(board)/self.game.getValidMoves(board).sum()
        probs = [x / counts_sum for x in counts]

        return probs
    
    def act(self, s, valids):
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        return best_act
    
    def updateQ(self, s, a, v):
        if(s[a] != '0'):
            print(s)
            raise ValueError("Updating Q function incorrectly: " + s[a])
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1

    def search(self, board, last_played, i=0):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """
        if(self.game.getGameEnded(board)):
            #If terminal return 1 as reward
            return 1
        
        valid = self.game.getValidMoves(board)
        b1, b2 = self.game.getCanonicalForm(board)
        s1, s2 = self.game.stringRepresentation(b1), self.game.stringRepresentation(b2) 

        if s1 not in self.Ps:
            # leaf node
            self.Ps[s1], v = self.nnet.predict(b1)
            self.Ps[s1] = self.Ps[s1] * valid # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s1])
            if sum_Ps_s <= 0:
                raise ValueError("Sum of Ps in MCTS should be more than 0")
            self.Ps[s1] /= sum_Ps_s  # renormalize
            self.Ns[s1] = 0

            self.Ps[s2], v = self.nnet.predict(b2)
            self.Ps[s2] = self.Ps[s2] * valid  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s2])
            if sum_Ps_s <= 0:
                raise ValueError("Sum of Ps in MCTS should be more than 0")
            self.Ps[s2] /= sum_Ps_s  # renormalize
            self.Ns[s2] = 0
            a1, a2 = self.act(s1, valid), self.act(s2, valid)
            assert s1[a1] == '0' and s2[a2] == '0'
            _, who_played = self.game.getNextState(board, a1, a2)
            return v if last_played == who_played else -v
        
        a1, a2 = self.act(s1, valid), self.act(s2, valid)
        next_s, who_played = self.game.getNextState(board, a1, a2) 
        print("Depth = ", i)                       
        v = self.search(next_s, who_played, i+1)
        if(who_played == 1):
            self.updateQ(s1, a1, v)
        else:
            self.updateQ(s2, a2, v)
        return v if last_played == who_played else -v
    
