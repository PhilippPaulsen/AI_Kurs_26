import streamlit as st
import numpy as np
import random
import time
import pandas as pd

# --- 1. CONFIG & CSS (Terminal Look) ---
st.set_page_config(page_title="AI2026 LAB", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
/* TERMINAL AESTHETICS */
@import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap');

html, body, [class*="st-"] {
    background-color: #000000;
    color: #e0e0e0;
    font-family: 'Roboto Mono', monospace;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0a0a0a;
    border-right: 1px solid #333;
}

/* Grid Container */
.grid-container {
    font-family: 'Courier New', monospace;
    font-size: 28px;
    line-height: 28px;
    white-space: pre;
    text-align: center;
    background-color: #050505;
    padding: 20px;
    border: 1px solid #333;
    border-radius: 5px;
    margin: 20px 0;
}

/* Didactic Box */
.theory-box {
    border: 1px dashed #00ff00;
    padding: 15px;
    margin-bottom: 20px;
    color: #00ff00;
    background-color: #001100;
}
.theory-title {
    font-weight: bold;
    border-bottom: 1px solid #00ff00;
    margin-bottom: 10px;
}

/* Buttons */
button {
    border-radius: 0 !important;
    border: 1px solid #555 !important;
    background-color: #111 !important;
    color: #fff !important;
}
button:hover {
    border-color: #fff !important;
    background-color: #222 !important;
}
</style>
""", unsafe_allow_html=True)

# --- 2. LOGIC: ENVIRONMENT & AGENTS ---

class Environment:
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.reset()
    
    def reset(self):
        # 0: Empty, 1: Wall, 2: Goal
        self.grid = np.zeros((self.height, self.width), dtype=int)
        self.start_pos = (1, 1)
        self.agent_pos = self.start_pos
        self.goal_pos = (self.height - 2, self.width - 2)
        
        # Walls (Random simple maze)
        blocks = int(self.width * self.height * 0.2)
        for _ in range(blocks):
            r, c = random.randint(0, self.height-1), random.randint(0, self.width-1)
            if (r, c) not in [self.start_pos, self.goal_pos]:
                self.grid[r, c] = 1
        
        # Ensure Goal
        self.grid[self.goal_pos] = 2
        
        self.visited = {self.start_pos}
        self.game_over = False
        
        # Model-Based Memory (Internal Map: 0=Unknown, 1=Wall, 2=Empty/Goal)
        self.memory_map = {} 

    def step(self, action):
        if self.game_over: return self.agent_pos, 0, True
        
        moves = [(-1,0), (1,0), (0,-1), (0,1)] # Up, Down, Left, Right
        dr, dc = moves[action]
        nr, nc = self.agent_pos[0]+dr, self.agent_pos[1]+dc
        
        # Bounds & Walls
        if 0 <= nr < self.height and 0 <= nc < self.width:
            cell = self.grid[nr, nc]
            if cell == 1: # Wall
                # Update memory if bumped
                self.memory_map[(nr, nc)] = 1 
                return self.agent_pos, -1, False
            else:
                self.agent_pos = (nr, nc)
                self.visited.add(self.agent_pos)
                
                # Check Goal
                if cell == 2:
                    self.game_over = True
                    return self.agent_pos, 100, True
                
                return self.agent_pos, -0.1, False
        
        return self.agent_pos, -1, False # Out of bounds

    def get_fog_view(self, radius=2):
        view = set()
        r, c = self.agent_pos
        for i in range(-radius, radius+1):
            for j in range(-radius, radius+1):
                if abs(i) + abs(j) <= radius + 1: # Manhattan-ish circle
                    view.add((r+i, c+j))
        return view

class QAgent:
    def __init__(self, env, alpha, gamma, epsilon):
        self.q = np.zeros((env.height, env.width, 4))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.prev_s = None
        self.prev_a = None
        
    def act(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        return np.argmax(self.q[s[0], s[1]])

    def learn(self, s, r, s_next):
        if self.prev_s is None: return
        
        old_q = self.q[self.prev_s[0], self.prev_s[1], self.prev_a]
        next_max = np.max(self.q[s_next[0], s_next[1]])
        
        new_q = old_q + self.alpha * (r + self.gamma * next_max - old_q)
        self.q[self.prev_s[0], self.prev_s[1], self.prev_a] = new_q
        
    def post_step(self, s, a):
        self.prev_s = s
        self.prev_a = a

# --- 3. STATE INITIALIZATION ---
if 'env' not in st.session_state:
    st.session_state.env = Environment(10, 10)
    st.session_state.agent_str = "Manual"
    st.session_state.logs = []
    st.session_state.q_agent = None

# --- 4. SIDEBAR ---
st.sidebar.title("LAB CONTROL")

# Grid Resizer
grid_n = st.sidebar.slider("Grid Size N", 5, 20, 10)
if grid_n != st.session_state.env.width:
    st.session_state.env = Environment(grid_n, grid_n)
    st.session_state.q_agent = None # Reset Q

# Agent Select
agent_type = st.sidebar.selectbox("Agent Intelligence", ["Manual", "Simple Reflex", "Model-based", "Q-Learning"])
st.session_state.agent_str = agent_type

# Q-Params (Context Sensitive)
alpha, gamma, epsilon = 0.5, 0.9, 0.1
if agent_type == "Q-Learning":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Hyperparameters")
    alpha = st.sidebar.slider("Alpha (Learning Rate)", 0.1, 1.0, 0.5)
    gamma = st.sidebar.slider("Gamma (Discount)", 0.1, 1.0, 0.9)
    epsilon = st.sidebar.slider("Epsilon (Exploration)", 0.0, 1.0, 0.1)

if st.sidebar.button("RESET SIMULATION"):
    st.session_state.env = Environment(grid_n, grid_n)
    st.session_state.q_agent = None
    st.session_state.logs = []
    st.rerun()

# --- 5. MAIN LOOP & LOGIC ---

env = st.session_state.env

# Didactic Theory Box
st.markdown('<div class="theory-box">', unsafe_allow_html=True)
if agent_type == "Manual":
    st.markdown('<div class="theory-title">MODE: MANUAL CONTROL</div>', unsafe_allow_html=True)
    st.write("You are the agent. Use the Arrow Keys to navigate the fog. Observe how partial observability affects your pathfinding.")
elif agent_type == "Simple Reflex":
    st.markdown('<div class="theory-title">MODE: SIMPLE REFLEX</div>', unsafe_allow_html=True)
    st.image(r"https://latex.codecogs.com/png.latex?\color{green}\text{Action}(p) = \text{Rules}[\text{State}(p)]")
    st.write("This agent **reacts** only to the immediate cell via hardcoded rules (If Wall -> Turn). It has NO memory and NO plan. It typically fails in loops.")
elif agent_type == "Model-based":
    st.markdown('<div class="theory-title">MODE: MODEL-BASED REFLEX</div>', unsafe_allow_html=True)
    st.write("This agent maintains an **Internal State** ($S'$). As it explores, it updates its private map of the world, effectively 'clearing' the fog in its mind, even if the sensors can't see far.")
elif agent_type == "Q-Learning":
    st.markdown('<div class="theory-title">MODE: Q-LEARNING (Reinforcement)</div>', unsafe_allow_html=True)
    st.latex(r"Q(s,a) \leftarrow Q(s,a) + \alpha [R + \gamma \max_{a'} Q(s',a') - Q(s,a)]")
    st.write(f"The agent learns the **Utility** of actions. $\\alpha={alpha}$ determines how fast new info overrides old. $\\gamma={gamma}$ determines future-sightedness.")
st.markdown('</div>', unsafe_allow_html=True)

# Keyboard Logic (JavaScript Injection)
# This script listens for keypresses and simulates clicks on the visible buttons below.
import streamlit.components.v1 as components

js_code = """
<script>
const doc = window.parent.document;
doc.addEventListener('keydown', function(e) {
    // Map keys to button text content (partial match)
    let btnToken = null;
    if (e.key === 'ArrowUp') btnToken = "UP";
    else if (e.key === 'ArrowDown') btnToken = "DOWN";
    else if (e.key === 'ArrowLeft') btnToken = "LEFT";
    else if (e.key === 'ArrowRight') btnToken = "RIGHT";
    
    if (btnToken) {
        const buttons = Array.from(doc.querySelectorAll('button'));
        const targetBtn = buttons.find(el => el.innerText.includes(btnToken));
        if (targetBtn) {
            targetBtn.click();
        }
    }
});
</script>
"""
# Inject the script (invisible)
components.html(js_code, height=0, width=0)

st.write("Controls: Use **Arrow Keys** or Buttons below.")

action = None # Initialize action to avoid NameError

if agent_type == "Manual":
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("UP ⬆️"): action = 0
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("LEFT ⬅️"): action = 2
    with col2:
        if st.button("DOWN ⬇️"): action = 1
    with col3:
        if st.button("RIGHT ➡️"): action = 3

elif agent_type != "Manual":
    if st.button("STEP / RUN"):
        if not env.game_over:
            
            # 1. REFLEX
            if agent_type == "Simple Reflex":
                # Dumb rule: Random safe move, bias towards goal if visible
                view = env.get_fog_view(2)
                possibles = []
                for i, (dr, dc) in enumerate([(-1,0), (1,0), (0,-1), (0,1)]):
                    nr, nc = env.agent_pos[0]+dr, env.agent_pos[1]+dc
                    if (nr, nc) in view:
                        if env.grid[nr, nc] == 2: possibles.append((i, 100))
                        elif env.grid[nr, nc] == 1: possibles.append((i, -100))
                        else: possibles.append((i, 0))
                    else:
                        possibles.append((i, -10)) # Unknown is scary but necessary
                
                possibles.sort(key=lambda x: x[1], reverse=True)
                action = possibles[0][0]
            
            # 2. MODEL-BASED
            elif agent_type == "Model-based":
                # Update memory
                c_view = env.get_fog_view(2)
                for pos in c_view:
                    if 0 <= pos[0] < env.height and 0 <= pos[1] < env.width:
                         env.memory_map[pos] = env.grid[pos]
                
                # Plan: Go to nearest 'Unknown' or Goal
                # Simple heuristic: Pick neighbor that is effectively empty and least visited?
                # Random for now to demonstrate memory via visual map
                action = random.randint(0, 3)

            # 3. Q-LEARNING
            elif agent_type == "Q-Learning":
                if st.session_state.q_agent is None:
                    st.session_state.q_agent = QAgent(env, alpha, gamma, epsilon)
                
                qa = st.session_state.q_agent
                # Update params
                qa.alpha, qa.gamma, qa.epsilon = alpha, gamma, epsilon
                
                s = env.agent_pos
                action = qa.act(s)
                qa.post_step(s, action)

# EXECUTE ACTION
if action is not None and not env.game_over:
    next_s, r, done = env.step(action)
    
    # Post-Step Learning
    if agent_type == "Q-Learning" and st.session_state.q_agent:
        st.session_state.q_agent.learn(None, r, next_s) # s is stored in prev_s
    
    if done:
        st.balloons()
        st.session_state.logs.append("TERMINUS REACHED.")
    else:
        st.rerun()

# --- 6. RENDERER (ASCII/EMOJI) ---

visible_mask = env.get_fog_view(radius=2)
grid_str = ""

for r in range(env.height):
    row_str = ""
    for c in range(env.width):
        pos = (r, c)
        
        # Determine Visibility
        is_visible = pos in visible_mask
        
        # Fog Logic
        # True Fog: If not visible, show Fog char
        # Model Memory: If not visible but in memory, show 'Ghost' char?
        
        symbol = "░" # Base Fog
        
        if is_visible:
            if pos == env.agent_pos: symbol = "🤖"
            elif pos == env.goal_pos: symbol = "🏁"
            elif env.grid[r, c] == 1: symbol = "🧱"
            else: symbol = "·"
            
            # Update Memory for Model Based visualization
            if agent_type == "Model-based":
                env.memory_map[pos] = env.grid[r, c]
                
        else:
            # Not visible. Check Memory (only for Model based?)
            # Or simplified: All agents have memory visually for the USER, but only model-based ACTS on it?
            # Prompt said: "True Fog of War: Fields outside sight are ░."
            # "Model-based Memory: ...markiert."
            
            if agent_type == "Model-based" and pos in env.memory_map:
                val = env.memory_map[pos]
                if val == 1: symbol = "▒" # Ghost Wall
                elif val == 2: symbol = "⚐" # Ghost Goal
                else: symbol = " " # Explored Empty
        
        row_str += symbol + " "
    grid_str += row_str + "\n"

st.markdown(f'<div class="grid-container">{grid_str}</div>', unsafe_allow_html=True)

# Q-Value Heatmap (Subtitle)
if agent_type == "Q-Learning" and st.session_state.q_agent:
    st.write("### Knowledge Map (Max Q)")
    # Render small table/grid
    q_grid = np.max(st.session_state.q_agent.q, axis=2)
    st.dataframe(pd.DataFrame(q_grid).style.background_gradient(cmap="Greens", axis=None))
