import streamlit as st
import time
import pandas as pd
import numpy as np
import random

# --- 1. ENVIRONMENT CLASS ---
class Grid:
    def __init__(self, width, height, coverage=0.2):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=int)
        self.start_pos = (0, 0)
        self.agent_pos = self.start_pos
        self.goal_pos = (height - 1, width - 1)
        self.walls = []
        self._generate_walls(coverage)
        self.grid[self.goal_pos] = 2
        self.visited = set([self.start_pos])

    def _generate_walls(self, coverage):
        num_walls = int(self.width * self.height * coverage)
        for _ in range(num_walls):
            while True:
                r, c = random.randint(0, self.height - 1), random.randint(0, self.width - 1)
                if (r, c) != self.start_pos and (r, c) != self.goal_pos:
                    self.grid[r, c] = 1
                    break

    def reset(self):
        self.agent_pos = self.start_pos
        self.visited = set([self.start_pos])
        return self.agent_pos

    def step(self, action):
        r, c = self.agent_pos
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Up, Down, Left, Right
        dr, dc = moves[action]
        nr, nc = r + dr, c + dc

        if 0 <= nr < self.height and 0 <= nc < self.width:
            if self.grid[nr, nc] == 1: return self.agent_pos, -1, False
            self.agent_pos = (nr, nc)
            self.visited.add(self.agent_pos)
            if self.agent_pos == self.goal_pos: return self.agent_pos, 100, True
            return self.agent_pos, -0.1, False
        return self.agent_pos, -1, False

    def get_percept(self):
        r, c = self.agent_pos
        res = {}
        moves = {'up':(-1,0), 'down':(1,0), 'left':(0,-1), 'right':(0,1)}
        for k, (dr, dc) in moves.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.height and 0 <= nc < self.width:
                if self.grid[nr, nc] == 1: res[k] = 'wall'
                elif self.grid[nr, nc] == 2: res[k] = 'goal'
                else: res[k] = 'visited' if (nr, nc) in self.visited else 'empty'
            else: res[k] = 'limit'
        return res

# --- 2. AGENT CLASSES ---
class SimpleReflexAgent:
    def __init__(self): self.log = []
    def act(self, percept):
        for i, d in enumerate(['up', 'down', 'left', 'right']):
            if percept[d] == 'goal':
                self.log.append(f"Ziel erkannt: {d}!")
                return i
        safe = [i for i, d in enumerate(['up', 'down', 'left', 'right']) if percept[d] not in ['wall', 'limit']]
        return random.choice(safe) if safe else 0

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.prev_state = None
        self.prev_action = None
        self.log = []

    def get_q(self, state):
        if state not in self.q_table: self.q_table[state] = np.zeros(4)
        return self.q_table[state]

    def act(self, state):
        if random.random() < self.epsilon:
            action = random.randint(0, 3)
            self.log.append("Exploration")
        else:
            q = self.get_q(state)
            action = np.argmax(q)
            self.log.append(f"Exploitation (Best Q: {np.max(q):.2f})")
        self.prev_state, self.prev_action = state, action
        return action

    def learn(self, current_state, reward):
        if self.prev_state is not None:
            old_q = self.get_q(self.prev_state)[self.prev_action]
            next_max = np.max(self.get_q(current_state))
            new_q = old_q + self.alpha * (reward + self.gamma * next_max - old_q)
            self.q_table[self.prev_state][self.prev_action] = new_q

# --- 3. STREAMLIT UI ---
st.set_page_config(page_title="AI Agent Playground", layout="wide")

if 'env' not in st.session_state:
    st.session_state.env = Grid(10, 10)
    st.session_state.agent = SimpleReflexAgent()
    st.session_state.history = []

st.title("🤖 AI Agent Playground")

col1, col2 = st.columns([2, 1])

with col1:
    # Grid Rendering
    env = st.session_state.env
    display = ""
    for r in range(env.height):
        row_str = ""
        for c in range(env.width):
            if (r, c) == env.agent_pos: row_str += "🤖 "
            elif (r, c) == env.goal_pos: row_str += "🏁 "
            elif env.grid[r, c] == 1: row_str += "🧱 "
            else: row_str += "· "
        display += row_str + "\n"
    st.code(display, language="text")

    if st.button("Step"):
        if isinstance(st.session_state.agent, QLearningAgent):
            state = env.agent_pos
            action = st.session_state.agent.act(state)
            next_s, reward, done = env.step(action)
            st.session_state.agent.learn(next_s, reward)
        else:
            action = st.session_state.agent.act(env.get_percept())
            env.step(action)
        st.rerun()

with col2:
    st.write("### Agent Log")
    for l in st.session_state.history[-5:]:
        st.text(l)
    
    if st.sidebar.button("Switch to Q-Learning"):
        st.session_state.agent = QLearningAgent()
        st.rerun()