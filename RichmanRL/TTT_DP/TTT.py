from functools import partial
import numpy as np
from tqdm import tqdm

from env import EnvSpec, Env, EnvWithModel
from policy import Policy

from dp import value_iteration, value_prediction
from monte_carlo import off_policy_mc_prediction_ordinary_importance_sampling as mc_ois
from monte_carlo import off_policy_mc_prediction_weighted_importance_sampling as mc_wis
from n_step_bootstrap import off_policy_n_step_sarsa as nsarsa
from n_step_bootstrap import on_policy_n_step_td as ntd

class TTT(Env): # MDP introduced at Fig 5.4 in Sutton Book
        def __init__(self):
            env_spec=EnvSpec(3**9,18,1.)
            super().__init__(env_spec)
            self.reset()

        def checkWin(self, state, player):
            board = np.reshape(state, (3,3))
            mask = board==player
            out = mask.all(0).any() | mask.all(1).any()
            return out or np.diag(mask).all() or np.diag(mask[:,::-1]).all()

        def is_terminal(self, state):
            return self.checkWin(state, 1) or self.checkWin(state, 2) or np.count_nonzero(state) == 9

        def reset(self):
            self._state = np.zeros(9)
            return self._state
        
        def stateToNum(self, state):
            return np.dot(state, np.power(3, np.arange(9)))

        def generate_traj(self):
            self.reset()
            terminal = False
            traj = []
            while(not terminal):
                print(self._state)
                action = np.random.choice(np.where(self._state == 0)[0])
                new_state, reward, terminal = self.step(self._state, action)
                traj.append((self.stateToNum(self._state), action, reward, self.stateToNum(new_state)))
                self._state = new_state
            return traj
        
        def step(self, state, action):
            new_state = np.array(state)
            if(np.random.rand() < 0.5):
                new_state[action] = 1
                return new_state, 1 if self.checkWin(new_state, 1) else 0, self.is_terminal(new_state)
            new_state[action] = 2
            return new_state, -1 if self.checkWin(new_state, 2) else 0, self.is_terminal(new_state)

if __name__ == "__main__":

    env = TTT()
    trajs = [env.generate_traj() for i in range(10_000)]
    initV = np.zeros(3**9)
    on_policy_n_step_td(env.spec, trajs, 4, 0.1, initV)








