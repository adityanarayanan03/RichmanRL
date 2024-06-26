from functools import partial
import numpy as np
from tqdm import tqdm
import pygame
import sys
from .env import EnvSpec, Env, EnvWithModel

from .monte_carlo import off_policy_mc_prediction_ordinary_importance_sampling as mc_ois
from .monte_carlo import off_policy_mc_prediction_weighted_importance_sampling as mc_wis
from .n_step_bootstrap import off_policy_n_step_sarsa as nsarsa
from .n_step_bootstrap import on_policy_n_step_td
from RichmanRL.utils import Policy, pickle_policy, get_pickled_policy
from RichmanRL.envs.typing_utils import RichmanObservation

pygame.init()


class TTTPolicy(Policy):

    def __init__(self, V: np.array):
        super(TTTPolicy, self).__init__(None, 9)
        self.V = V
    
    def stateToNum(self, state):
        return int(np.dot(state, np.power(3, np.arange(9))))
    
    def flatten_board(self, board):
        flattened = np.zeros(9, dtype=np.int32)
        for i in range(3):
            for j in range(3):
                if board[i, j, 0] == 1:
                    flattened[i * 3 + j] = 1  # My piece
                elif board[i, j, 1] == 1:
                    flattened[i * 3 + j] = 2  # Opponent's piece
                else:
                    flattened[i * 3 + j] = 0  # Empty cell
        return flattened

    def __call__(self, state: RichmanObservation):
        board = self.flatten_board(state["observation"][2])
        valid_actions = np.where(board == 0)[0]
        action_values = []
        for index in valid_actions:
            board[index] = 1
            action_values.append((index, self.V[self.stateToNum(board)]))
            board[index] = 0
        
        return sorted(action_values, key= lambda x: x[1])[-1][0]
    
    def update(self):
        raise NotImplementedError("Cannot call update function")



class TTT(Env):  # MDP introduced at Fig 5.4 in Sutton Book
    def __init__(self):
        env_spec = EnvSpec(3**9, 18, 1.0)
        super().__init__(env_spec)
        self.reset()
        self.seen = set()

    def checkWin(self, state, player):
        board = np.reshape(state, (3, 3))
        mask = board == player
        out = mask.all(0).any() | mask.all(1).any()
        return out or np.diag(mask).all() or np.diag(mask[:, ::-1]).all()

    def is_terminal(self, state):
        return (
            self.checkWin(state, 1)
            or self.checkWin(state, 2)
            or np.count_nonzero(state) == 9
        )

    def reset(self):
        self._state = np.zeros(9)
        return self._state

    def stateToNum(self, state):
        return int(np.dot(state, np.power(3, np.arange(9))))

    def generate_traj(self):
        self.reset()
        terminal = False
        traj = []
        while not terminal:
            action = np.random.choice(np.where(self._state == 0)[0])
            new_state, reward, terminal = self.step(self._state, action)
            traj.append(
                (
                    self.stateToNum(self._state),
                    action,
                    reward,
                    self.stateToNum(new_state),
                )
            )
            self.seen.add(self.stateToNum(self._state))
            self._state = new_state
        self.seen.add(self.stateToNum(self._state))
        return traj

    def step(self, state, action):
        new_state = np.array(state)
        if np.random.rand() < 0.5:
            new_state[action] = 1
            return (
                new_state,
                1 if self.checkWin(new_state, 1) else 0,
                self.is_terminal(new_state),
            )
        new_state[action] = 2
        return (
            new_state,
            -1 if self.checkWin(new_state, 2) else 0,
            self.is_terminal(new_state),
        )


class BoardVisualizer:
    def __init__(self, value_function, seen):
        # Constants
        self.WIDTH, self.HEIGHT = 300, 300
        self.LINE_WIDTH = 5
        self.BOARD_SIZE = 3
        self.SQUARE_SIZE = self.WIDTH // self.BOARD_SIZE
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        # Initialize the screen
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Tic Tac Toe")
        # Font
        self.font = pygame.font.Font(None, 36)
        # Board state
        self.board = np.zeros(9)
        self.values = value_function
        self.seen = seen

    def draw_board(self):
        self.screen.fill(self.WHITE)
        for i in range(1, self.BOARD_SIZE):
            pygame.draw.line(
                self.screen,
                self.BLACK,
                (0, i * self.SQUARE_SIZE),
                (self.WIDTH, i * self.SQUARE_SIZE),
                self.LINE_WIDTH,
            )
            pygame.draw.line(
                self.screen,
                self.BLACK,
                (i * self.SQUARE_SIZE, 0),
                (i * self.SQUARE_SIZE, self.HEIGHT),
                self.LINE_WIDTH,
            )
        print(
            f"Current Board State Value: {self.values[self.get_base_3_num()]}, Seen: {self.get_base_3_num() in self.seen}"
        )
        for row in range(3):
            for col in range(3):
                self.draw_mark(row, col)

    def draw_mark(self, row, col):
        x = col * self.SQUARE_SIZE + self.SQUARE_SIZE // 2
        y = row * self.SQUARE_SIZE + self.SQUARE_SIZE // 2
        if self.board[3 * row + col] == 1:
            pygame.draw.line(
                self.screen,
                self.BLACK,
                (x - self.SQUARE_SIZE // 4, y - self.SQUARE_SIZE // 4),
                (x + self.SQUARE_SIZE // 4, y + self.SQUARE_SIZE // 4),
                self.LINE_WIDTH,
            )
            pygame.draw.line(
                self.screen,
                self.BLACK,
                (x + self.SQUARE_SIZE // 4, y - self.SQUARE_SIZE // 4),
                (x - self.SQUARE_SIZE // 4, y + self.SQUARE_SIZE // 4),
                self.LINE_WIDTH,
            )
        elif self.board[3 * row + col] == 2:
            pygame.draw.circle(
                self.screen, self.BLACK, (x, y), self.SQUARE_SIZE // 4, self.LINE_WIDTH
            )

    def get_base_3_num(self):
        return int(np.dot(self.board, np.power(3, np.arange(9))))

    def draw(self):
        # Game loop
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = pygame.mouse.get_pos()
                    row = y // self.SQUARE_SIZE
                    col = x // self.SQUARE_SIZE
                    self.board[3 * row + col] = (self.board[3 * row + col] + 1) % 3
                    self.draw_board()
            pygame.display.flip()

def get_policy():
    env = TTT()
    trajs = [env.generate_traj() for i in tqdm(range(1_000))]
    initV = np.zeros(3**9)
    V = on_policy_n_step_td(env.spec, trajs, 2, 0.3, initV)
    return TTTPolicy(initV)

if __name__ == "__main__":
    # env = TTT()
    # trajs = [env.generate_traj() for i in tqdm(range(2_000_000))]
    # initV = np.zeros(3**9)
    # V = on_policy_n_step_td(env.spec, trajs, 2, 0.3, initV)
    # b = BoardVisualizer(V, env.seen)
    # b.draw()
    # policy = TTTPolicy(initV)
    policy = get_pickled_policy("TTT_n_step_policy.pkl", "/home/anant/projects/RichmanRL")
    b = BoardVisualizer(policy.V, set())
    b.draw()

