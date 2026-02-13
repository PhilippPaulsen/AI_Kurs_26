import numpy as np
import random

class Grid:
    def __init__(self, width, height, coverage=0.2):
        self.width = width
        self.height = height
        # 0: Empty, 1: Wall, 2: Goal, 3: Agent, 4: Visited
        self.grid = np.zeros((height, width), dtype=int)
        self.start_pos = (0, 0)
        self.agent_pos = self.start_pos
        self.goal_pos = (height - 1, width - 1)
        self.walls = []
        self._generate_walls(coverage)
        self.grid[self.goal_pos] = 2
        self.visited = set()
        self.visited.add(self.start_pos)

    def _generate_walls(self, coverage):
        num_walls = int(self.width * self.height * coverage)
        for _ in range(num_walls):
            while True:
                r = random.randint(0, self.height - 1)
                c = random.randint(0, self.width - 1)
                if (r, c) != self.start_pos and (r, c) != self.goal_pos and self.grid[r, c] == 0:
                    self.grid[r, c] = 1
                    self.walls.append((r, c))
                    break

    def reset(self):
        self.agent_pos = self.start_pos
        self.visited = set()
        self.visited.add(self.start_pos)
        return self.agent_pos

    def step(self, action):
        """
        Executes an action.
        Actions: 0: Up, 1: Down, 2: Left, 3: Right
        Returns: next_state, reward, done
        """
        r, c = self.agent_pos
        if action == 0: # Up
            nr, nc = r - 1, c
        elif action == 1: # Down
            nr, nc = r + 1, c
        elif action == 2: # Left
            nr, nc = r, c - 1
        elif action == 3: # Right
            nr, nc = r, c + 1
        else:
            nr, nc = r, c

        # Check interaction
        if 0 <= nr < self.height and 0 <= nc < self.width:
            if self.grid[nr, nc] == 1: # Wall
                # Stay in place, negative reward
                return self.agent_pos, -1, False
            elif self.grid[nr, nc] == 2: # Goal
                self.agent_pos = (nr, nc)
                self.visited.add((nr, nc))
                return self.agent_pos, 100, True
            else: # Empty or Visited
                self.agent_pos = (nr, nc)
                self.visited.add((nr, nc))
                return self.agent_pos, -0.1, False # Small penalty for step
        else:
            # Out of bounds, stay in place
            return self.agent_pos, -1, False

    def get_percept(self, pos=None):
        if pos is None:
            pos = self.agent_pos
        r, c = pos
        surroundings = {}
        # 0: Up, 1: Down, 2: Left, 3: Right
        # Return content of adjacent cells
        # Content: 'limit', 'wall', 'empty', 'goal', 'visited'
        
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        keys = ['up', 'down', 'left', 'right']

        for i, (dr, dc) in enumerate(moves):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.height and 0 <= nc < self.width:
                val = self.grid[nr, nc]
                if val == 1:
                    surroundings[keys[i]] = 'wall'
                elif val == 2:
                    surroundings[keys[i]] = 'goal'
                elif (nr, nc) in self.visited:
                    surroundings[keys[i]] = 'visited' # For model-based
                else:
                    surroundings[keys[i]] = 'empty'
            else:
                surroundings[keys[i]] = 'limit'
        return surroundings
