from pettingzoo import AECEnv  # noqa: D100
import gymnasium
import numpy as np
from gymnasium import spaces
import pygame
from pettingzoo.utils import agent_selector, wrappers  # noqa: F401
import math

# Python3 program to implement Disjoint Set Data
# Structure.


class DisjSet:  # noqa: D101
    def __init__(self, n):  # noqa: D107
        self.rank = [1] * n
        self.parent = [i for i in range(n)]

    def find(self, x):  # noqa: D102
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):  # noqa: D102
        xset = self.find(x)
        yset = self.find(y)

        if xset == yset:
            return

        if self.rank[xset] < self.rank[yset]:
            self.parent[xset] = yset
        elif self.rank[xset] > self.rank[yset]:
            self.parent[yset] = xset
        else:
            self.parent[yset] = xset
            self.rank[xset] = self.rank[xset] + 1


class HexBoard:  # noqa: D101
    def __init__(self, size):  # noqa: D107
        self.size = size
        self.board = np.zeros((size, size))
        self.legal = np.ones(size * size)
        self.dsu = DisjSet(size * size + 4)
        self.PLAYER1 = 1
        self.PLAYER2 = 2

    def play_action(self, row, column, player):  # noqa: D102
        if player not in [self.PLAYER1, self.PLAYER2]:
            raise Exception(f"Invalid player passed in: {player}")
        if self.board[row][column] != 0:
            raise Exception(
                f"{row}, {column} is already taken by player {self.board[row][column]}"
            )
        if row < 0 or row >= self.size or column < 0 or column >= self.size:
            raise Exception("Row or column is out of bounds")

        self.board[row][column] = player
        self.legal[self.nodeToInt(row, column)] = 0
        neighbors = self.getNeighbors(row, column)
        # Union find with neighbors
        for x, y in neighbors:
            if self.board[x][y] == player:
                self.dsu.union(self.nodeToInt(x, y), self.nodeToInt(row, column))
        # Update win condition checks
        if row == 0 and player == self.PLAYER1:
            self.dsu.union(self.nodeToInt(row, column), self.size * self.size)
        if row == self.size - 1 and player == self.PLAYER1:
            self.dsu.union(self.nodeToInt(row, column), self.size * self.size + 1)
        if column == 0 and player == self.PLAYER2:
            self.dsu.union(self.nodeToInt(row, column), self.size * self.size + 2)
        if column == self.size - 1 and player == self.PLAYER2:
            self.dsu.union(self.nodeToInt(row, column), self.size * self.size + 3)

    def nodeToInt(self, row, column):  # noqa: D102
        return self.size * row + column

    def getNeighbors(self, row, column):  # noqa: D102
        potentialNeighbors = [
            (row, column - 1),
            (row, column + 1),
            (row - 1, column),
            (row - 1, column + 1),
            (row + 1, column),
            (row + 1, column - 1),
        ]
        ret = []
        for x, y in potentialNeighbors:
            if x < 0 or x >= self.size or y < 0 or y >= self.size:
                continue
            ret.append((x, y))
        return ret

    def win(self, player):  # noqa: D102
        if player == self.PLAYER1:
            return self.dsu.find(self.size * self.size) == self.dsu.find(
                self.size * self.size + 1
            )
        elif player == self.PLAYER2:
            return self.dsu.find(self.size * self.size + 2) == self.dsu.find(
                self.size * self.size + 3
            )
        raise Exception("Invalid player inputted for win check")

    def check_game_over(self):  # noqa: D102
        return self.win(self.PLAYER1) or self.win(self.PLAYER2)


class Hex(AECEnv):  # noqa: D101
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "hex_v0",
        "is_parallelizable": False,
        "render_fps": 50,
    }

    def __init__(self, board_size=11, render_mode: str | None = None):  # noqa: D107
        super().__init__()
        self.board = HexBoard(board_size)

        self.agents = ["player_1", "player_2"]
        self.possible_agents = self.agents[:]
        self.board_size = board_size

        self.action_spaces = {i: spaces.Discrete(board_size**2) for i in self.agents}
        self.observation_spaces = {
            i: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0, high=1, shape=(board_size, board_size, 2), dtype=np.int8
                    ),
                    "action_mask": spaces.Box(
                        low=0, high=1, shape=(board_size**2,), dtype=np.int8
                    ),
                }
            )
            for i in self.agents
        }

        self.rewards = {i: 0 for i in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {
            i: {"legal_moves": list(range(0, board_size**2))} for i in self.agents
        }

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.render_mode = render_mode
        self.screen_height = board_size * 50
        self.screen = None

        if self.render_mode == "human":
            self.clock = pygame.time.Clock()

    def observe(self, agent):  # noqa: D102
        board_vals = np.array(self.board.board)
        cur_player = self.possible_agents.index(agent)
        opp_player = (cur_player + 1) % 2

        cur_p_board = np.equal(board_vals, cur_player + 1)
        opp_p_board = np.equal(board_vals, opp_player + 1)

        observation = np.stack([cur_p_board, opp_p_board], axis=2).astype(np.int8)
        legal_moves = (
            np.array(self.board.legal)
            #if agent == self.agent_selection
            #else np.zeros(self.board_size**2)
        )

        return {"observation": observation, "action_mask": legal_moves}

    def observation_space(self, agent):  # noqa: D102
        return self.observation_spaces[agent]

    def action_space(self, agent):  # noqa: D102
        return self.action_spaces[agent]

    # action in this case is a value from 0 to 8 indicating position to move on tictactoe board  # noqa: E501

    def step(self, action):  # noqa: D102
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            return self._was_dead_step(action)
        # play turn
        row, column = action // self.board_size, action % self.board_size
        self.board.play_action(row, column, self.agents.index(self.agent_selection) + 1)

        # update infos
        # list of valid actions (indexes in board)
        # next_agent = self.agents[(self.agents.index(self.agent_selection) + 1) % len(self.agents)]  # noqa: E501
        next_agent = self._agent_selector.next()

        if self.board.check_game_over():
            if self.board.win(1):
                # agent 0 won
                self.rewards[self.agents[0]] += 1
                self.rewards[self.agents[1]] -= 1
            elif self.board.win(2):
                # agent 1 won
                self.rewards[self.agents[1]] += 1
                self.rewards[self.agents[0]] -= 1
            else:
                raise Exception("There was a tie??")
            # once either play wins or there is a draw, game over, both players are done
            self.terminations = {i: True for i in self.agents}

        # Switch selection to next agents
        self._cumulative_rewards[self.agent_selection] = 0
        self.agent_selection = next_agent

        self._accumulate_rewards()
        if self.render_mode == "human":
            self.render()

    def reset(self, seed=None, options=None):  # noqa: D102
        # reset environment
        self.board = HexBoard(self.board_size)

        self.agents = self.possible_agents[:]
        self.rewards = {i: 0 for i in self.agents}
        self._cumulative_rewards = {i: 0 for i in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}
        # selects the first agent
        self._agent_selector.reinit(self.agents)
        self._agent_selector.reset()
        self.agent_selection = self._agent_selector.reset()

        if self.screen is None:
            pygame.init()

        if self.render_mode == "human":
            self.screen = pygame.display.set_mode(
                (self.screen_height * 1.5, self.screen_height)
            )
            pygame.display.set_caption("Hex")
        else:
            self.screen = pygame.Surface((self.screen_height * 1.5, self.screen_height))

    def close(self):  # noqa: D102
        pass

    def render(self):  # noqa: D102
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        screen_height = self.screen_height
        hex_radius = screen_height / self.board_size / 2
        WHITE = (255, 255, 255)
        RED = (255, 0, 0)
        BLUE = (0, 255, 0)
        colors = [WHITE, RED, BLUE]

        for row in range(self.board_size):
            for col in range(self.board_size):
                x = 2 * hex_radius * col + hex_radius * row + hex_radius
                y = hex_radius + row * hex_radius * math.sqrt(3)
                pygame.draw.circle(
                    self.screen,
                    colors[int(self.board.board[row][col])],
                    (x, y),
                    hex_radius,
                )

        if self.render_mode == "human":
            pygame.display.update()
            
            if self.board.legal.sum() == 0 and not self.board.check_game_over():
                while(1):
                    pass
            
            self.clock.tick(self.metadata["render_fps"])
            

        observation = np.array(pygame.surfarray.pixels3d(self.screen))

        return (
            np.transpose(observation, axes=(1, 0, 2))
            if self.render_mode == "rgb_array"
            else None
        )
