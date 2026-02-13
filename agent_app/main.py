import streamlit as st
import time
import pandas as pd
import numpy as np
import random

# --- 1. ENVIRONMENT CLASS ---
class Grid:
    def __init__(self, width, height, coverage=0.2):
        self.width, self.height = width, height
        self.reset(coverage)

    def reset(self, coverage=0.2):
        self.grid = np.zeros((self.height, self.width), dtype=int)
        self.start_pos = (0, 0)
        self.agent_pos = self.start_pos
        self.goal_pos = (self.height - 1, self.width - 1)
        self.visited = {self.start_pos}
        # Walls
        num_walls = int(self.width * self.height * coverage)
        for _ in range(num_walls):
            r, c = random.randint(0, self.height-1), random.randint(0, self.width-1)
            if (r, c) not in [self.start_pos, self.goal_pos]:
                self.grid[r, c] = 1
        return self.agent_pos

    def step(self, action):
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)] # 0:Up, 1:Down, 2:Left, 3:Right
        r, c = self.agent_pos
        dr, dc = moves[action]
        nr, nc = r + dr, c + dc

        if 0 <= nr < self.height and 0 <= nc < self.width and self.grid[nr, nc] != 1:
            self.agent_pos = (nr, nc)
            self.visited.add(self.agent_pos)
            if self.agent_pos == self.goal_pos: return self.agent_pos, 100, True
            return self.agent_pos, -0.1, False
        return self.agent_pos, -0.7, False # Wandtreffer bestrafen

    def get_percept(self):
        r, c = self.agent_pos
        p = {}
        for i, d in enumerate(['up', 'down', 'left', 'right']):
            dr, dc = [(-1,0), (1,0), (0,-1), (0,1)][i]
            nr, nc = r+dr, c+dc
            if not (0 <= nr < self.height and 0 <= nc < self.width): p[d] = 'limit'
            elif self.grid[nr, nc] == 1: p[d] = 'wall'
            elif (nr, nc) == self.goal_pos: p[d] = 'goal'
            else: p[d] = 'visited' if (nr, nc) in self.visited else 'empty'
        return p

# --- 2. AGENT CLASSES ---
class SimpleReflexAgent:
    def __init__(self): self.log = []
    def act(self, percept):
        for i, d in enumerate(['up', 'down', 'left', 'right']):
            if percept[d] == 'goal': return i
        safe = [i for i, d in enumerate(['up', 'down', 'left', 'right']) if percept[d] not in ['wall', 'limit']]
        return random.choice(safe) if safe else 0

class QLearningAgent:
    def __init__(self, alpha=0.5, gamma=0.9, epsilon=0.1):
        self.q_table, self.log = {}, []
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.prev_state = self.prev_action = None

    def get_q(self, state):
        if state not in self.q_table: self.q_table[state] = np.zeros(4)
        return self.q_table[state]

    def act(self, state):
        self.prev_state = state
        if random.random() < self.epsilon: self.prev_action = random.randint(0, 3)
        else: self.prev_action = np.argmax(self.get_q(state))
        return self.prev_action

    def learn(self, current_state, reward):
        if self.prev_state is None: return
        q = self.get_q(self.prev_state)
        q[self.prev_action] += self.alpha * (reward + self.gamma * np.max(self.get_q(current_state)) - q[self.prev_action])

# --- 3. UI ---
st.set_page_config(page_title="AI Agent Playground", layout="wide")
st.title("🤖 AI Agent Playground")

if 'env' not in st.session_state:
    st.session_state.env = Grid(8, 8)
    st.session_state.agent = SimpleReflexAgent()
    st.session_state.history = []

col1, col2 = st.columns([2, 1])

with col1:
    env = st.session_state.env
    grid_view = ""
    for r in range(env.height):
        row = ""
        for c in range(env.width):
            if (r, c) == env.agent_pos: row += "🤖 "
            elif (r, c) == env.goal_pos: row += "🏁 "
            elif env.grid[r, c] == 1: row += "🧱 "
            else: row += "· "
        grid_view += row + "\n"
    st.code(grid_view, language="text")

    if st.button("Simulations-Schritt"):
        agent = st.session_state.agent
        if isinstance(agent, QLearningAgent):
            state = env.agent_pos
            action = agent.act(state)
            next_s, reward, done = env.step(action)
            agent.learn(next_s, reward)
            st.session_state.history.append(f"Q-Step: {state} -> {reward}")
        else:
            action = agent.act(env.get_percept())
            env.step(action)
            st.session_state.history.append("Reflex Move")
        if env.agent_pos == env.goal_pos: st.balloons()
        st.rerun()

with col2:
    mode = st.radio("Agenten-Typ wählen:", ["Reflex", "Q-Learning"])
    if st.button("Typ übernehmen & Reset"):
        st.session_state.env.reset()
        st.session_state.agent = QLearningAgent() if mode == "Q-Learning" else SimpleReflexAgent()
        st.session_state.history = []
        st.rerun()
    
    st.write("### Log")
    for msg in st.session_state.history[-5:]: st.text(msg)