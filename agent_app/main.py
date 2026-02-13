import streamlit as st
import numpy as np
import random
import time
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(page_title="AI 2026: AGENT TERMINAL", layout="wide", initial_sidebar_state="expanded")

# Retro Terminal CSS
st.markdown("""
<style>
/* Global Terminal Look */
.stApp {
    background-color: #0e0e0e;
    color: #00ff00;
    font-family: 'Courier New', Courier, monospace;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #1a1a1a;
    border-right: 1px solid #333;
}
[data-testid="stSidebar"] * {
    color: #00ff00 !important;
    font-family: 'Courier New', Courier, monospace !important;
}

/* Monospace Grid Container */
.terminal-grid {
    font-family: 'Courier New', Courier, monospace;
    white-space: pre;
    font-size: 24px;
    line-height: 24px;
    background-color: #111;
    color: #00ff00;
    padding: 20px;
    border: 2px solid #00ff00;
    border-radius: 5px;
    text-align: center;
    box-shadow: 0 0 10px rgba(0, 255, 0, 0.2);
    margin-bottom: 20px;
}

/* Log Monitor */
.terminal-log {
    font-family: 'Courier New', Courier, monospace;
    font-size: 14px;
    color: #00cc00;
    background-color: #000;
    border: 1px solid #333;
    padding: 10px;
    height: 300px;
    overflow-y: auto;
    border-left: 3px solid #00ff00;
}

/* Headers */
h1, h2, h3 {
    color: #00ff00 !important;
    text-transform: uppercase;
    text-shadow: 0 0 5px #00ff00;
}

/* Buttons */
.stButton>button {
    background-color: #003300;
    color: #00ff00;
    border: 1px solid #00ff00;
    font-family: 'Courier New', Courier, monospace;
}
.stButton>button:hover {
    background-color: #00ff00;
    color: #000;
}

</style>
""", unsafe_allow_html=True)

# --- 2. LOGIC: ENVIRONMENT ---

class Environment:
    def __init__(self, level="Open Field", width=12, height=12):
        self.width = width
        self.height = height
        self.level_name = level
        self.reset()

    def reset(self):
        # 0: Empty, 1: Wall, 2: Goal, 3: Pit/Trap
        self.grid = np.zeros((self.height, self.width), dtype=int)
        self.start_pos = (1, 1)
        self.agent_pos = self.start_pos
        self.goal_pos = (self.height - 2, self.width - 2)
        self.visited = {self.start_pos}
        self.game_over = False
        self.win = False

        self._generate_level(self.level_name)
    
    def _generate_level(self, level):
        # Borders
        for r in range(self.height):
            self.grid[r, 0] = 1
            self.grid[r, self.width-1] = 1
        for c in range(self.width):
            self.grid[0, c] = 1
            self.grid[self.height-1, c] = 1
            
        if level == "The Maze":
            # Simple recursive-like walls or random blocks
            for _ in range(int(self.width * self.height * 0.2)):
                r, c = random.randint(1, self.height-2), random.randint(1, self.width-2)
                if (r, c) not in [self.start_pos, self.goal_pos]:
                    self.grid[r, c] = 1
                    
        elif level == "Stochastic Trap":
            # Add some walls
            for _ in range(int(self.width * self.height * 0.15)):
                r, c = random.randint(1, self.height-2), random.randint(1, self.width-2)
                if (r, c) not in [self.start_pos, self.goal_pos]:
                    self.grid[r, c] = 1
            # Add Traps (invisible in simple view, or clearly marked?)
            # Let's mark them as 3 (Pit)
            for _ in range(3):
                r, c = random.randint(1, self.height-2), random.randint(1, self.width-2)
                if (r, c) not in [self.start_pos, self.goal_pos] and self.grid[r, c] == 0:
                    self.grid[r, c] = 3

        # Set Goal
        self.grid[self.goal_pos] = 2

    def step(self, action):
        """
        Action: 0:Up, 1:Down, 2:Left, 3:Right
        Returns: next_state (pos), reward, done
        """
        if self.game_over: return self.agent_pos, 0, True

        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Up, Down, Left, Right
        dr, dc = moves[action]
        r, c = self.agent_pos
        nr, nc = r + dr, c + dc
        
        # Collision Check
        cell_type = self.grid[nr, nc]
        
        if cell_type == 1: # Wall
            return self.agent_pos, -0.5, False # Bump penalty
        
        self.agent_pos = (nr, nc)
        self.visited.add(self.agent_pos)
        
        if cell_type == 2: # Goal
            self.game_over = True
            self.win = True
            return self.agent_pos, 100, True
        elif cell_type == 3: # Pit
            self.game_over = True
            self.win = False
            return self.agent_pos, -50, True # Death penalty
        else:
            return self.agent_pos, -0.1, False # Living penalty

    def get_view(self, radius=2):
        """Returns a dict of visible cells for Fog of War."""
        visible = {}
        r, c = self.agent_pos
        for i in range(-radius, radius+1):
            for j in range(-radius, radius+1):
                nr, nc = r+i, c+j
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    visible[(nr, nc)] = self.grid[nr, nc]
        return visible

# --- 3. LOGIC: AGENTS ---

class Agent:
    def __init__(self): self.log = []
    def act(self, percept): raise NotImplementedError

class ManualAgent(Agent):
    def act(self, percept, user_action=None):
        return user_action

class ReflexAgent(Agent):
    def act(self, percept):
        # Percept is a dict of nearby cells
        # Need to parse immediate neighbors
        # We need the relative direction from the agent's POV in the percept?
        # Envs 'get_view' returns absolute coords. Agent needs to map to actions.
        # Let's cheat slightly and pass the env or use a standard sensor model.
        # Simplified: Percept = { 'up': 'wall', 'down': 'empty', ... }
        
        # Map absolute view to relative
        pass # implemented in main logic loop to keep class simple or pass full state

class QLearningAgent(Agent):
    def __init__(self, width, height, alpha=0.5, gamma=0.9, epsilon=0.1):
        super().__init__()
        self.q_table = np.zeros((height, width, 4)) # H, W, Actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.prev_state = None
        self.prev_action = None

    def act(self, state_pos):
        r, c = state_pos
        if random.random() < self.epsilon:
            action = random.randint(0, 3)
            self.log.append("EXPLORE (Random)")
        else:
            action = np.argmax(self.q_table[r, c])
            self.log.append(f"EXPLOIT (Q={self.q_table[r, c, action]:.2f})")
        
        self.prev_state = state_pos
        self.prev_action = action
        return action

    def learn(self, current_state_pos, reward):
        if self.prev_state is None: return
        
        pr, pc = self.prev_state
        cr, cc = current_state_pos
        
        old_q = self.q_table[pr, pc, self.prev_action]
        next_max = np.max(self.q_table[cr, cc])
        
        new_q = old_q + self.alpha * (reward + self.gamma * next_max - old_q)
        self.q_table[pr, pc, self.prev_action] = new_q

# --- 4. STREAMLIT APP STATE ---

if 'env' not in st.session_state:
    st.session_state.env = Environment()
    st.session_state.agent_type = "Manual"
    st.session_state.q_agent = QLearningAgent(12, 12)
    st.session_state.logs = ["SYSTEM INITIALIZED..."]
    st.session_state.step = 0
    st.session_state.total_reward = 0

# --- 5. SIDEBAR CONTROLS ---

st.sidebar.title("COMMAND CENTER")
level_sel = st.sidebar.selectbox("SELECT LEVEL", ["Open Field", "The Maze", "Stochastic Trap"])
agent_sel = st.sidebar.selectbox("SELECT AGENT", ["Manual", "Reflex", "Model-based", "Q-Learning"])

st.sidebar.markdown("---")
st.sidebar.markdown("**HYPERPARAMETERS**")
alpha = st.sidebar.slider("ALPHA_LC (α)", 0.0, 1.0, 0.5)
gamma = st.sidebar.slider("GAMMA_DF (γ)", 0.0, 1.0, 0.9)
epsilon = st.sidebar.slider("EPSILON_XP (ε)", 0.0, 1.0, 0.1)
speed = st.sidebar.slider("SIM_SPEED (sec)", 0.0, 1.0, 0.1)

# Reset Logic
if st.sidebar.button("SYSTEM RESET") or level_sel != st.session_state.env.level_name:
    st.session_state.env = Environment(level=level_sel)
    st.session_state.q_agent = QLearningAgent(12, 12, alpha, gamma, epsilon) # Reset Q or keep? Let's reset for new level.
    st.session_state.logs = ["SYSTEM PROCESSED RESET COMMAND."]
    st.session_state.step = 0
    st.session_state.total_reward = 0
    st.rerun()

# Update Live Params
st.session_state.q_agent.alpha = alpha
st.session_state.q_agent.gamma = gamma
st.session_state.q_agent.epsilon = epsilon
st.session_state.agent_type = agent_sel

# --- 6. MAIN LOGIC LOOP ---

env = st.session_state.env

def game_step(action):
    # Execute
    state_before = env.agent_pos
    next_state, reward, done = env.step(action)
    
    # Learn
    if st.session_state.agent_type == "Q-Learning":
        st.session_state.q_agent.learn(next_state, reward)
        log_msg = st.session_state.q_agent.log[-1] if st.session_state.q_agent.log else "Learning..."
    else:
        log_msg = f"Moved {['NORTH','SOUTH','WEST','EAST'][action]}"

    # Update State
    st.session_state.step += 1
    st.session_state.total_reward += reward
    st.session_state.logs.append(f"STEP {st.session_state.step:03d}: {log_msg} | Rwd: {reward}")
    
    if done:
        res = "MISSION SUCCESS" if env.win else "MISSION FAILED"
        st.session_state.logs.append(f"*** {res} *** Total Reward: {st.session_state.total_reward}")

# UI Layout
col_term, col_dsh = st.columns([2, 1])

with col_term:
    st.title(f"// {level_sel} //")
    
    # --- KEYBOARD INPUT (JS INJECTION) ---
    # Hidden input trick or Components? Using simple buttons is safer but requested Keys.
    # We will use st.components.v1.html to capture keypresses and set query params or generic logic?
    # Actually, let's stick to buttons/auto-run for robustness, or use a known tricky method.
    # For now, let's provide on-screen DPAD for Manual, and AUTO for others.
    
    # RENDER GRID
    # We build the ASCII string based on View
    radius = 2 # Fog radius
    view = env.get_view(radius)
    
    grid_str = ""
    for r in range(env.height):
        row_str = ""
        for c in range(env.width):
            # VISIBILITY LOGIC
            visible = False
            if st.session_state.agent_type == "Model-based":
                if (r, c) in env.visited or (r,c) in view: visible = True
            elif st.session_state.agent_type == "Manual": # God mode or Fog? Let's do Fog
                if (r, c) in view or (r,c) in env.visited: visible = True
            else: # Q-Learning / Reflex can often 'see' or we show God mode for teaching? 
                # Request says "Fog of War"
                 if abs(r - env.agent_pos[0]) <= radius and abs(c - env.agent_pos[1]) <= radius: visible = True
                 if (r, c) in env.visited: visible = True # Show visited trail?
            
            # OVERRIDE FOR DEBUG/Q-LEARNING? 
            # If Q-Learning, maybe show Q-Values overlay instead of Fog?
            # Let's keep Fog for "Agent Perspective" but maybe dim it.
            
            # ASCII CHARS
            # ░ ▒ ▓ █
            val = env.grid[r, c]
            
            if (r, c) == env.agent_pos:
                char = "A "
            elif (r, c) == env.goal_pos:
                char = "G " if visible else "? "
            elif val == 1:
                char = "##" if visible else ".." # ??
            elif val == 3:
                char = "XX" if visible else ".."
            else:
                # Empty
                if not visible:
                    char = "░░"
                else:
                    # If Q-Learning, show arrow/value?
                    if st.session_state.agent_type == "Q-Learning":
                        # Show max action?
                        best_a = np.argmax(st.session_state.q_agent.q_table[r, c])
                        # ^ v < >
                        arrows = ["^ ", "v ", "< ", "> "]
                        # Only show if visited/learned?
                        if np.max(st.session_state.q_agent.q_table[r, c]) != 0:
                            char = arrows[best_a]
                        else:
                            char = ". "
                    else:
                        char = ". "
            
            row_str += char
        grid_str += row_str + "\n"
    
    st.markdown(f'<div class="terminal-grid">{grid_str}</div>', unsafe_allow_html=True)
    
    # MANUAL CONTROLS
    if st.session_state.agent_type == "Manual" and not env.game_over:
        try:
            from streamlit_keyup import keyup
            # "debounce" helps to avoid too many rapid triggers, "key_handling='stop'" stops propagation
            key = keyup("manual_keys", label="Keyboard Control (Focus here)", key_handling="stop", debounce=50)
        except ImportError:
            st.warning("Install 'streamlit-keyup' for keyboard support. using fallback buttons.")
            key = None

        # Process Key
        # We need to ensure we don't re-process the same key event infinitely if we don't clear it?
        # Streamlit execution model: 
        # 1. User presses Key inside keyup component -> Component value changes -> Rerun
        # 2. Script runs, `key` is "ArrowUp".
        # 3. We execute step.
        # 4. We assume user releases or presses another key.
        # Problem: If user presses ArrowUp, `key` stays "ArrowUp" for subsequent reruns (e.g. if we press a button elsewhere)?
        # For now, let's assume the keyup component allows consecutive presses.
        # To strictly avoid double-moves on non-key reruns, we might need to track 'last_processed_key_id' or similar, 
        # but keyup usually doesn't provide a unique ID per press.
        # However, `game_step` updates state. If we just run it, it's fine for the immediate reaction.
        
        if key == "ArrowUp": 
            game_step(0)
            st.rerun() # Force update to clear/reset if needed or just show result
        elif key == "ArrowDown": 
            game_step(1)
            st.rerun()
        elif key == "ArrowLeft": 
            game_step(2)
            st.rerun()
        elif key == "ArrowRight": 
            game_step(3)
            st.rerun()
        
        st.markdown("<small>Click the box above to use Arrow Keys ⬆️⬇️⬅️➡️</small>", unsafe_allow_html=True)

        c1, c2, c3, c4, c5 = st.columns(5)
        with c2: 
            if st.button("UP"): game_step(0); st.rerun()
        with c4:
             pass
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: 
            if st.button("LEFT"): game_step(2); st.rerun()
        with c2: 
            if st.button("DOWN"): game_step(1); st.rerun()
        with c3: 
            if st.button("RIGHT"): game_step(3); st.rerun()

    # AUTO CONTROLS & AGENT LOGIC
    if st.session_state.agent_type != "Manual":
        if st.checkbox("INIT_AUTO_SEQUENCE", value=False):
            if not env.game_over:
                act = 0
                
                # --- AGENT DECISION LOGIC ---
                if st.session_state.agent_type == "Q-Learning":
                    act = st.session_state.q_agent.act(env.agent_pos)
                
                elif st.session_state.agent_type == "Reflex":
                    # Smart Reflex: Avoid Walls, Seek Goal
                    # Get percepts
                    possible_moves = [] # (action, score)
                    dirs = [(-1,0), (1,0), (0,-1), (0,1)]
                    
                    for i, (dr, dc) in enumerate(dirs):
                        nr, nc = env.agent_pos[0]+dr, env.agent_pos[1]+dc
                        if 0 <= nr < env.height and 0 <= nc < env.width:
                            cell = env.grid[nr, nc]
                            if cell == 2: # Goal
                                possible_moves.append((i, 100))
                            elif cell == 1: # Wall
                                possible_moves.append((i, -100))
                            elif cell == 3: # Pit
                                possible_moves.append((i, -50))
                            else:
                                possible_moves.append((i, 0)) # Neutral
                        else:
                            possible_moves.append((i, -100)) # Bound
                    
                    # Sort by score
                    possible_moves.sort(key=lambda x: x[1], reverse=True)
                    # Pick best, if tie pick random among best
                    best_score = possible_moves[0][1]
                    best_actions = [x[0] for x in possible_moves if x[1] == best_score]
                    act = random.choice(best_actions)
                    
                    st.session_state.logs.append(f"REFLEX: Scanned neighbors. Best Action: {act} (Score {best_score})")

                elif st.session_state.agent_type == "Model-based":
                    # Random Exploration + Memory (Avoid Loops)
                    # Prefer unvisited cells
                    dirs = [(-1,0), (1,0), (0,-1), (0,1)]
                    unvisited = []
                    visited_safe = []
                    
                    for i, (dr, dc) in enumerate(dirs):
                        nr, nc = env.agent_pos[0]+dr, env.agent_pos[1]+dc
                        if 0 <= nr < env.height and 0 <= nc < env.width and env.grid[nr, nc] != 1:
                            if (nr, nc) not in env.visited:
                                unvisited.append(i)
                            else:
                                visited_safe.append(i)
                    
                    if unvisited:
                        act = random.choice(unvisited)
                        st.session_state.logs.append("MODEL: Found unvisited cell. Exploring.")
                    elif visited_safe:
                        act = random.choice(visited_safe)
                        st.session_state.logs.append("MODEL: All known neighbors visited. Backtracking.")
                    else:
                        act = random.randint(0, 3) # Stuck?
                
                game_step(act)
                time.sleep(speed)
                st.rerun()

with col_dsh:
    st.markdown("### > SYSTEM_LOG")
    # Limit log size
    if len(st.session_state.logs) > 20:
        st.session_state.logs = st.session_state.logs[-20:]
        
    log_txt = "\n".join(st.session_state.logs[::-1])
    st.markdown(f'<div class="terminal-log">{log_txt}</div>', unsafe_allow_html=True)
    
    st.metric("STEP_COUNT", st.session_state.step)
    st.metric("NET_REWARD", f"{st.session_state.total_reward:.1f}")