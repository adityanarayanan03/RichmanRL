from __future__ import annotations
from RichmanRL.envs import HexBoard
from RichmanRL.utils import Policy
import numpy as np

class HexPolicy(Policy):

    def __init__(self):
        self.seen_states = {}
    
    def hash(self, state):
        return ''.join([str(x) for x in state.flatten()])

    def __call__(self, state: RichmanObservation):
        base_board = state["observation"][2][:,:,0] + 2*state["observation"][2][:,:,1]
        if(self.hash(base_board) in self.seen_states):
            return self.seen_states[self.hash(base_board)]
        mask = base_board != 0
        losing_square = np.zeros((11,11))
        NUM_TIMES = 200
        for i in range(NUM_TIMES):
            random_moves = np.random.randint(1,3, (11,11))
            random_moves[mask] = base_board[mask]
            board = HexBoard(11)
            for i in range(11):
                for j in range(11):
                    board.play_action(i, j, random_moves[i,j])
            assert np.allclose(board.board, random_moves)
            assert board.win(1) ^ board.win(2)
            if(board.win(1)):
                losing_mask = board.board == 2
            else:
                losing_mask = board.board == 1
            losing_square[losing_mask] += 1
        # Create a masked array using the subset mask
        masked_board = np.ma.masked_array(losing_square, mask=mask)
        lossPercent, index = masked_board.min()/NUM_TIMES, masked_board.argmin()
        self.seen_states[self.hash(base_board)] = (lossPercent, index)
        assert state["action_mask"][1][index] == 1
        return self.seen_states[self.hash(base_board)]
    
    def update(self):
        pass

class HexGamePolicy(Policy):

    def __init__(self, base_policy):
        self.base_policy = base_policy
    
    def __call__(self, state: RichmanObservation):
        return self.base_policy(state)[1]
    
    def update(self, *args, **kwargs):
        pass

class HexBiddingPolicy(Policy):

    def __init__(self, base_policy):
        self.base_policy = base_policy
    
    def __call__(self, state: RichmanObservation):
        return int((0.5 - self.base_policy(state)[0])*state["observation"][0])
    
    def update(self, *args, **kwargs):
        pass






