import random
import numpy as np

class Agent:
    def __init__(self):
        self.log = []

    def act(self, percept):
        raise NotImplementedError

    def reset(self):
        self.log = []

class ManualAgent(Agent):
    def act(self, percept, action_intent=None):
        if action_intent is not None:
             self.log.append(f"User pressed button for action {action_intent}.")
             return action_intent
        return None

class SimpleReflexAgent(Agent):
    def act(self, percept):
        # Rule 1: If goal is adjacent, move there
        for action, direction in enumerate(['up', 'down', 'left', 'right']):
            if percept[direction] == 'goal':
                self.log.append(f"SAW GOAL at {direction}! Moving there.")
                return action
        
        # Rule 2: If no goal, pick a random safe move
        safe_moves = []
        for action, direction in enumerate(['up', 'down', 'left', 'right']):
            if percept[direction] not in ['wall', 'limit']:
                safe_moves.append(action)
        
        if safe_moves:
            action = random.choice(safe_moves)
            dirs = ['Up', 'Down', 'Left', 'Right']
            self.log.append(f"No immediate goal. Randomly chose safe move: {dirs[action]}.")
            return action
        else:
            self.log.append("Trapped! No safe moves.")
            return random.choice([0,1,2,3]) # Desperation move

class ModelBasedReflexAgent(Agent):
    def __init__(self):
        super().__init__()
        self.visited = set()
        
    def reset(self):
        super().reset()
        self.visited = set()

    def update_internal_model(self, pos):
        self.visited.add(pos)

    def act(self, percept):
        # Rule 1: Goal
        for action, direction in enumerate(['up', 'down', 'left', 'right']):
            if percept[direction] == 'goal':
                self.log.append(f"Memory: SAW GOAL at {direction}!")
                return action
        
        # Rule 2: Prefer unvisited cells
        unvisited_safe_moves = []
        visited_safe_moves = []
        
        for action, direction in enumerate(['up', 'down', 'left', 'right']):
            status = percept[direction]
            if status == 'empty': # unvisited in percept context usually implies not visited? 
                # Note: 'empty' in percept just means 0 in grid. 
                # The agent needs to track its own visited state or rely on environment enrichment.
                # Here we assume the agent uses its own memory based on action result, 
                # BUT since 'act' doesn't get current pos, we might need to pass it or rely on percept 'visited'.
                # The environment's 'get_percept' marks 'visited' if it's in env.visited. 
                # Let's assume the agent trusts the percept 'visited' tag for simplicity 
                # OR we implement proper state tracking.
                # Let's use the percept's 'visited' tag which comes from env.
                if status != 'visited': 
                     unvisited_safe_moves.append(action)
            elif status == 'visited':
                visited_safe_moves.append(action)
        
        if unvisited_safe_moves:
            action = random.choice(unvisited_safe_moves)
            dirs = ['Up', 'Down', 'Left', 'Right']
            self.log.append(f"Exploring new territory: {dirs[action]}.")
            return action
        elif visited_safe_moves:
             action = random.choice(visited_safe_moves)
             dirs = ['Up', 'Down', 'Left', 'Right']
             self.log.append(f"Backtracking/Wandering visited area: {dirs[action]}.")
             return action
        else:
            return random.choice([0,1,2,3])

class QLearningAgent(Agent):
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        super().__init__()
        self.q_table = {} # Key: (state), Value: [q_up, q_down, q_left, q_right]
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.prev_state = None
        self.prev_action = None

    def get_q(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(4)
        return self.q_table[state]

    def reset(self):
        super().reset()
        self.prev_state = None
        self.prev_action = None
        # Do not reset Q-table to keep learning

    def act(self, state): # Here 'percept' is the full state coordinate for Q-learning
        # Epsilon-greedy
        if random.uniform(0, 1) < self.epsilon:
            action = random.randint(0, 3)
            self.log.append("Exploring (Random Action)")
        else:
            q_values = self.get_q(state)
            # Pick max, break ties randomly
            max_q = np.max(q_values)
            actions_with_max_q = [i for i, q in enumerate(q_values) if q == max_q]
            action = random.choice(actions_with_max_q)
            self.log.append(f"Exploiting (Best Q: {max_q:.2f})")
        
        self.prev_state = state
        self.prev_action = action
        return action

    def learn(self, current_state, reward):
        if self.prev_state is not None:
             old_q = self.get_q(self.prev_state)[self.prev_action]
             next_max_q = np.max(self.get_q(current_state))
             
             new_q = old_q + self.alpha * (reward + self.gamma * next_max_q - old_q)
             self.q_table[self.prev_state][self.prev_action] = new_q
             
             # self.log.append(f"Updated Q({self.prev_state}, {self.prev_action}) to {new_q:.2f}")
