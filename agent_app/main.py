import streamlit as st
import numpy as np
import random
import time
import pandas as pd

# --- 1. CONFIG & CSS (Terminal Look) ---
st.set_page_config(page_title="KI-Labor 2026", layout="wide", initial_sidebar_state="expanded")

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

/* Compact Manual Control Buttons */
div[data-testid="column"] button {
    padding: 0.2rem 0.5rem !important;
    font-size: 0.8rem !important;
    min-height: 0px !important;
}
</style>
""", unsafe_allow_html=True)

# --- 2. LOGIC: ENVIRONMENT & AGENTS ---

class Environment:
    def __init__(self, width=10, height=10, step_penalty=-0.1, goal_reward=100.0):
        self.width = width
        self.height = height
        self.step_penalty = step_penalty
        self.goal_reward = goal_reward
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
                    return self.agent_pos, self.goal_reward, True
                
                return self.agent_pos, self.step_penalty, False
        
        return self.agent_pos, -1, False # Out of bounds

    def get_percept_view(self, radius=1):
        view = set()
        r, c = self.agent_pos
        for i in range(-radius, radius+1):
            for j in range(-radius, radius+1):
                if abs(i) + abs(j) <= radius: # Strict Manhattan distance
                    view.add((r+i, c+j))
        return view
        
    def get_current_percept_text(self):
        # Didactic: Textual description of immediate neighbors
        r, c = self.agent_pos
        percepts = {}
        # Order: Up, Down, Left, Right
        directions = {(-1,0): "UP", (1,0): "DOWN", (0,-1): "LEFT", (0,1): "RIGHT"}
        
        for (dr, dc), label in directions.items():
            nr, nc = r+dr, c+dc
            if 0 <= nr < self.height and 0 <= nc < self.width:
                val = self.grid[nr, nc]
                if val == 1: percepts[label] = "WALL"
                elif val == 2: percepts[label] = "GOAL"
                else: percepts[label] = "EMPTY"
            else:
                percepts[label] = "BOUNDARY"
        return percepts

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
            # st.session_state.logs.append(f"🎲 **Zufalls-Zug** (Exploration) bei {s}")
            return random.randint(0, 3)
        
        vals = self.q[s[0], s[1]]
        max_v = np.max(vals)
        action = np.argmax(vals)
        # st.session_state.logs.append(f"🧠 **Gieriger Zug** (Exploitation) bei {s} (Q={max_v:.2f})")
        return action

    def learn(self, s, r, s_next):
        if self.prev_s is None: return
        
        old_q = self.q[self.prev_s[0], self.prev_s[1], self.prev_a]
        next_max = np.max(self.q[s_next[0], s_next[1]])
        
        new_q = old_q + self.alpha * (r + self.gamma * next_max - old_q)
        self.q[self.prev_s[0], self.prev_s[1], self.prev_a] = new_q
        
        # Didactic Log (Only log occasionally or detailed mode?)
        # actions = ["OBEN", "UNTEN", "LINKS", "RECHTS"]
        # act_str = actions[self.prev_a]
        # log_msg = (f"🎯 **Q-Update** @ {self.prev_s} Aktion {act_str}:<br>"
        #            f"`Q_neu = {old_q:.2f} + {self.alpha} * ({r:.2f} + {self.gamma} * {next_max:.2f} - {old_q:.2f})` "
        #            f"➡️ **{new_q:.3f}**")
        # st.session_state.logs.append(log_msg)
        
    def post_step(self, s, a):
        self.prev_s = s
        self.prev_a = a

# --- 3. STATE INITIALIZATION ---
if 'env' not in st.session_state:
    st.session_state.env = Environment(10, 10, step_penalty=-0.1, goal_reward=100.0)
    st.session_state.agent_str = "Manuell"
    st.session_state.logs = []
    st.session_state.q_agent = None
    
    # Performance Stats (Per Agent Type)
    st.session_state.stats = {
        "Manuell": {'wins': 0, 'episodes': 0, 'total_steps': 0, 'total_return': 0.0},
        "Reflex-Agent": {'wins': 0, 'episodes': 0, 'total_steps': 0, 'total_return': 0.0},
        "Modell-basiert": {'wins': 0, 'episodes': 0, 'total_steps': 0, 'total_return': 0.0},
        "Q-Learning": {'wins': 0, 'episodes': 0, 'total_steps': 0, 'total_return': 0.0}
    }
    # Current Episode Tracker
    st.session_state.current_episode = {'steps': 0, 'return': 0.0}
    # Training History for Q-Learning
    st.session_state.training_history = []

# --- 4. SIDEBAR ---
st.sidebar.title("LABOR STEUERUNG (v1.3)")

# Grid Resizer
grid_n = st.sidebar.slider("Gittergröße N", 5, 20, 10)
if grid_n != st.session_state.env.width:
    st.session_state.env = Environment(grid_n, grid_n)
    st.session_state.q_agent = None # Reset Q
    st.session_state.training_history = []

# Agent Select
agent_type = st.sidebar.selectbox("Agenten Intelligenz", ["Manuell", "Reflex-Agent", "Modell-basiert", "Q-Learning"])
st.session_state.agent_str = agent_type

# Q-Params (Context Sensitive)
alpha, gamma, epsilon = 0.5, 0.9, 0.1
if agent_type == "Q-Learning":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Hyperparameter (Lernen)")
    alpha = st.sidebar.slider("Alpha (Lernrate)", 0.1, 1.0, 0.5, help="Wie stark neue Informationen alte Informationen überschreiben.")
    gamma = st.sidebar.slider("Gamma (Diskount)", 0.1, 1.0, 0.9, help="Wie wichtig zukünftige Belohnungen gegenüber sofortigen sind.")
    epsilon = st.sidebar.slider("Epsilon (Exploration)", 0.0, 1.0, 0.1, help="Wahrscheinlichkeit für zufällige Aktionen (Exploration) vs. beste bekannte Aktion (Exploitation).")
    
    if st.sidebar.button("Train Episodes (50x)"):
        # Training Loop
        progress_bar = st.sidebar.progress(0)
        
        # Ensure Agent Exists
        if st.session_state.q_agent is None:
             st.session_state.q_agent = QAgent(st.session_state.env, alpha, gamma, epsilon)
        
        qa = st.session_state.q_agent
        # Update params
        qa.alpha, qa.gamma, qa.epsilon = alpha, gamma, epsilon
        
        for ep in range(50):
            # Reset Environment but KEEP Q-Table
            # We need to reuse the existing environment but 'clean' the grid positions? 
            # Or make a new env with same dimensions?
            # Better: Reset current env to start pos.
            st.session_state.env.reset()
            # Note: Env.reset() randomizes walls too in current implementation.
            # If we want to learn ONE maze, we should separate 'reset_pos' from 'reset_maze'.
            # For this simple app, re-randomizing maze makes it harder to learn unless map is static.
            # Let's Modify Env.reset() logic? 
            # THE USER REQUEST didn't specify static maze. 
            # But Q-learning on RANDOM mazes requires state input to include local view, currently state is (r,c).
            # If maze changes, (r,c) meaning changes. 
            # IMPLICIT REQUIREMENT for convergence: Maze must be static during training episodes OR state representation must be relative.
            # For this level of course (AI_Kurs_26), let's make RESET keep the walls if we are training?
            # Or just let it be. 
            # Wait, standard Gridworld Q-Learning (Table based on Coord) FAILS if walls move.
            # So I should PROBABLY keep the walls fixed for the "Train Episodes" loop.
            
            # Temporary fix: Don't randomize walls on simple reset if we want to learn 'this' maze?
            # Current Env.reset() randomizes walls.
            # Let's stick to the requested task: "Q-Tabelle darf ... nicht gelöscht werden".
            # If the maze changes, the Q-table (Layout based) becomes invalid.
            # So effectively we should NOT re-randomize walls every episode for Q-Learning on coordinates.
            # I will modify Environment.reset to have an option to keep layout.
            pass
            
            # ACTUALLY: The user didn't ask to fix the maze. But if I don't, the graph won't converge. 
            # I will assume for "Train Episodes", we are training on the CURRENT maze.
            # But `env.reset()` re-gens walls.
            # I will perform a 'soft reset' manually here.
            
            env = st.session_state.env
            env.agent_pos = env.start_pos
            env.game_over = False
            env.visited = {env.start_pos}
            # Keep walls!
            
            steps = 0
            ep_return = 0
            qa.prev_s = None # Reset previous state for new episode
            
            # Run Episode
            while not env.game_over and steps < 200: # Limit steps
                s = env.agent_pos
                a = qa.act(s)
                qa.post_step(s, a)
                
                next_s, r, done = env.step(a)
                qa.learn(s, r, next_s)
                
                steps += 1
                ep_return += r
            
            # Log Stat
            st.session_state.training_history.append(steps)
            progress_bar.progress((ep + 1) / 50)
        
        st.session_state.logs.append(f"✅ **Training abgeschlossen:** 50 Episoden auf aktueller Karte.")
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("Simulations-Steuerung")
auto_run = st.sidebar.checkbox("Auto-Lauf (Simulation)", value=False)
speed = st.sidebar.slider("Geschwindigkeit (Wartezeit in s)", 0.0, 1.0, 0.2)
percept_enabled = st.sidebar.checkbox("Percept Field (Sichtfeld)", value=True, help="Wenn aktiv, sieht der Agent nur benachbarte Felder (Radius 1).")

# NEW: Current Percept Expander
with st.sidebar.expander("👁️ Current Percept", expanded=True):
    percepts = st.session_state.env.get_current_percept_text()
    # Compact 2-Column Layout
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"⬆️ `{percepts['UP']}`")
        st.write(f"⬅️ `{percepts['LEFT']}`")
    with col2:
        st.write(f"⬇️ `{percepts['DOWN']}`")
        st.write(f"➡️ `{percepts['RIGHT']}`")

st.sidebar.markdown("---")
st.sidebar.subheader("Umgebungs-Konfiguration")
env_step_penalty = st.sidebar.slider("Schritt-Strafe (Kosten)", -2.0, 0.0, -0.1, 0.1, help="Kosten (negativ) für jeden Schritt.")
env_goal_reward = st.sidebar.slider("Ziel-Belohnung", 10.0, 200.0, 100.0, 10.0, help="Belohnung für das Erreichen des Ziels.")

if st.sidebar.button("RESET SIMULATION"):
    # Full Reset including Walls
    st.session_state.env = Environment(grid_n, grid_n, step_penalty=env_step_penalty, goal_reward=env_goal_reward)
    st.session_state.q_agent = None
    st.session_state.logs = []
    st.session_state.current_episode = {'steps': 0, 'return': 0.0}
    st.session_state.training_history = []
    st.rerun()

# Stats Display
st.sidebar.markdown("---")
with st.sidebar.expander("📊 Leistungs-Statistik", expanded=False):
    # Convert stats to DataFrame for nice display
    stats_data = []
    for ag, data in st.session_state.stats.items():
        episodes = data['episodes']
        if episodes > 0:
            win_rate = (data['wins'] / episodes) * 100
            avg_steps = data['total_steps'] / episodes
            avg_return = data['total_return'] / episodes
        else:
            win_rate, avg_steps, avg_return = 0.0, 0.0, 0.0
            
        stats_data.append({
            "Agent": ag,
            "Siegrate": f"{win_rate:.1f}%",
            "Ø Schritte": f"{avg_steps:.1f}",
            "Ø Return": f"{avg_return:.2f}"
        })
    
    st.dataframe(pd.DataFrame(stats_data).set_index("Agent"), use_container_width=True)
    if st.button("Statistik zurücksetzen"):
        for ag in st.session_state.stats:
             st.session_state.stats[ag] = {'wins': 0, 'episodes': 0, 'total_steps': 0, 'total_return': 0.0}
        st.rerun()

# --- 5. HELPER FUNCTIONS ---

def render_grid_html(env, agent_type, percept_enabled, q_agent=None):
    # Radius 1 for Percept
    visible_mask = env.get_percept_view(radius=1) if percept_enabled else {(r,c) for r in range(env.height) for c in range(env.width)}
    
    # Calculate Max Q for normalization if needed
    max_q_val = 1.0
    if q_agent:
        max_q_val = np.max(q_agent.q) if np.max(q_agent.q) > 0 else 1.0

    grid_str = ""
    for r in range(env.height):
        row_str = ""
        for c in range(env.width):
            pos = (r, c)
            is_visible = pos in visible_mask
            
            # Base styles
            style = ""
            symbol = "░" # Unobserved
            
            q_color = None
            if q_agent:
                q_vals = q_agent.q[r, c]
                best_q = np.max(q_vals)
                if best_q != 0:
                    intensity = int(min(255, max(50, (abs(best_q) / max_q_val) * 200))) 
                    if best_q < -0.1: # Negative
                         q_color = f"rgba(255, 0, 0, 0.3)"
                    elif best_q > 0.1: # Positive
                         q_color = f"rgba(0, {intensity}, 0, 0.5)"

            # Determine Symbol
            if is_visible:
                # Update Model Memory if Model-based
                if agent_type == "Modell-basiert": 
                    env.memory_map[pos] = env.grid[r, c]

                if pos == env.agent_pos: symbol = "🤖"
                elif pos == env.goal_pos: symbol = "🏁"
                elif env.grid[r, c] == 1: symbol = "🧱"
                else: symbol = "·"
                
                # Apply Q-color background if visible
                if q_color: style = f"background-color: {q_color};"

            else:
                # In Percept Shadow (Unobserved)
                symbol = "░" 
                style = "color: #333;" # Dimmed
                
                # Check Memory for Model-Based
                if agent_type == "Modell-basiert" and pos in env.memory_map:
                    val = env.memory_map[pos]
                    if val == 1: symbol = "▒" # Ghost Wall
                    elif val == 2: symbol = "⚐" # Ghost Goal
                    else: symbol = "&nbsp;" # Empty Known
                    style = "color: #555;" # Dimmed for memory
                
                # Q-Learning: Show known Q-values even in shadow?
                # User constraint: "Felder außerhalb ... visuell als 'Unobserved' (░) markiert werden."
                # But seeing the Q-Table grow is nice.
                # Let's faintly show Q-color behind the fog symbol if known
                elif q_agent and np.max(q_agent.q[r, c]) != 0:
                     if q_color: style = f"background-color: {q_color}; color: #555;"

            # Construct Cell HTML
            if style:
                row_str += f'<span style="{style}">{symbol}</span> '
            else:
                row_str += symbol + " "
                
        grid_str += row_str + "\n"
    
    return f'<div class="grid-container">{grid_str}</div>'

# --- 5. MAIN LOOP & LOGIC ---

env = st.session_state.env

# Didactic Theory Box
st.markdown('<div class="theory-box">', unsafe_allow_html=True)
if agent_type == "Manuell":
    st.markdown('<div class="theory-title">MODUS: MANUELLE STEUERUNG</div>', unsafe_allow_html=True)
    st.write("Du bist der Agent. Das **Percept Field** ist dein Sichtbereich (Radius 1). Außerhalb davon ist alles 'Unobserved' (░).")
elif agent_type == "Reflex-Agent":
    st.markdown('<div class="theory-title">MODUS: REFLEX-AGENT (Einfach)</div>', unsafe_allow_html=True)
    st.image(r"https://latex.codecogs.com/png.latex?\color{green}\text{Aktion}(p) = \text{Regel}[\text{Sensor}(p)]")
    st.write("Dieser Agent reagiert nur auf sein **Percept**. Er hat KEIN Gedächtnis.")
elif agent_type == "Modell-basiert":
    st.markdown('<div class="theory-title">MODUS: MODELL-BASIERTER REFLEX-AGENT</div>', unsafe_allow_html=True)
    st.write("Dieser Agent speichert beobachtete Felder in einer **Internal Map**. Er erinnert sich an Wände und besuchte Orte, auch wenn sie nicht mehr im Percept sind.")
elif agent_type == "Q-Learning":
    st.markdown('<div class="theory-title">MODUS: Q-LEARNING (Verstärkendes Lernen)</div>', unsafe_allow_html=True)
    st.write("Der Agent lernt durch **Episoden**. Nutze den 'Train Episodes' Button, um das Lernen zu beschleunigen. Beobachte das Konvergenz-Diagramm unten.")
st.markdown('</div>', unsafe_allow_html=True)

# Keyboard Logic (JavaScript Injection)
import streamlit.components.v1 as components

js_code = """
<script>
const doc = window.parent.document;
doc.addEventListener('keydown', function(e) {
    let btnToken = null;
    if (e.key === 'ArrowUp') btnToken = "⬆️";
    else if (e.key === 'ArrowDown') btnToken = "⬇️";
    else if (e.key === 'ArrowLeft') btnToken = "⬅️";
    else if (e.key === 'ArrowRight') btnToken = "➡️";
    
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
components.html(js_code, height=0, width=0)

st.write("Steuerung: Nutze **Pfeiltasten** oder Buttons.")

# Live Metrics
curr_ep = st.session_state.current_episode
col_m1, col_m2 = st.columns(2)
col_m1.metric("Schritte (Aktuell)", curr_ep['steps'])
col_m2.metric("Return (Aktuell)", f"{curr_ep['return']:.2f}")

action = None # Initialize action

# MANUAL INPUT MAPPING
if agent_type == "Manuell":
    # Compact Centered Layout
    _, col_up, _ = st.columns([14, 2, 14])
    with col_up:
        if st.button("⬆️", key="btn_up_v2", help="Nach Oben"): action = 0
            
    _, col_left, col_down, col_right, _ = st.columns([12, 2, 2, 2, 12])
    with col_left:
        if st.button("⬅️", key="btn_left_v2", help="Nach Links"): action = 2
    with col_down:
        if st.button("⬇️", key="btn_down_v2", help="Nach Unten"): action = 1
    with col_right:
        if st.button("➡️", key="btn_right_v2", help="Nach Rechts"): action = 3

# AUTOMATIC AGENT LOGIC
if agent_type != "Manuell":
    steps_to_run = 0
    if st.button("SCHRITT / RUN"):
        steps_to_run = 1
    elif auto_run:
        steps_to_run = 50 
    
    if steps_to_run > 0 and not env.game_over:
        placeholder = st.empty()
        
        for _ in range(steps_to_run):
            if env.game_over: break
            
            # 1. REFLEX
            if agent_type == "Reflex-Agent":
                view = env.get_percept_view(1) # Uses Percept
                possibles = []
                for i, (dr, dc) in enumerate([(-1,0), (1,0), (0,-1), (0,1)]):
                    nr, nc = env.agent_pos[0]+dr, env.agent_pos[1]+dc
                    
                    if 0 <= nr < env.height and 0 <= nc < env.width:
                        if (nr, nc) in view:
                            if env.grid[nr, nc] == 2: possibles.append((i, 100))
                            elif env.grid[nr, nc] == 1: possibles.append((i, -100))
                            else: possibles.append((i, 0))
                        else:
                            possibles.append((i, -10)) # Unknown
                    else:
                        possibles.append((i, -100)) # Out of bounds
                
                max_score = max(possibles, key=lambda x: x[1])[1]
                best_moves = [move for move, score in possibles if score == max_score]
                action = random.choice(best_moves)
            
            # 2. MODEL-BASED
            elif agent_type == "Modell-basiert":
                c_view = env.get_percept_view(1)
                for pos in c_view:
                    if 0 <= pos[0] < env.height and 0 <= pos[1] < env.width:
                         env.memory_map[pos] = env.grid[pos]
                action = random.randint(0, 3)

            # 3. Q-LEARNING
            elif agent_type == "Q-Learning":
                if st.session_state.q_agent is None:
                    st.session_state.q_agent = QAgent(env, alpha, gamma, epsilon)
                
                qa = st.session_state.q_agent
                qa.alpha, qa.gamma, qa.epsilon = alpha, gamma, epsilon
                
                s = env.agent_pos
                action = qa.act(s)
                qa.post_step(s, action)

            # EXECUTE ACTION
            if action is not None and not env.game_over:
                next_s, r, done = env.step(action)
                
                st.session_state.current_episode['steps'] += 1
                st.session_state.current_episode['return'] += r
                
                if agent_type == "Q-Learning" and st.session_state.q_agent:
                    st.session_state.q_agent.learn(None, r, next_s)
                
                if steps_to_run > 1:
                     grid_html = render_grid_html(env, agent_type, percept_enabled, st.session_state.q_agent if agent_type == "Q-Learning" else None)
                     placeholder.markdown(grid_html, unsafe_allow_html=True)
                     time.sleep(speed)
                     
                     if done: 
                         st.balloons()
                         st.session_state.logs.append("ZIEL ERREICHT!")
                         
                         stat_entry = st.session_state.stats[agent_type]
                         stat_entry['episodes'] += 1
                         stat_entry['wins'] += 1
                         stat_entry['total_steps'] += st.session_state.current_episode['steps']
                         stat_entry['total_return'] += st.session_state.current_episode['return']
                         break
                         
        if auto_run and not env.game_over:
             time.sleep(speed)
             st.rerun()

# MANUAL EXECUTION
if agent_type == "Manuell" and action is not None and not env.game_over:
    next_s, r, done = env.step(action)
    
    st.session_state.current_episode['steps'] += 1
    st.session_state.current_episode['return'] += r
    
    if done:
        st.balloons()
        st.session_state.logs.append("ZIEL ERREICHT!")
        
        stat_entry = st.session_state.stats[agent_type]
        stat_entry['episodes'] += 1
        stat_entry['wins'] += 1
        stat_entry['total_steps'] += st.session_state.current_episode['steps']
        stat_entry['total_return'] += st.session_state.current_episode['return']

# --- 6. RENDERER (ASCII/EMOJI) ---
if not (agent_type != "Manuell" and auto_run):
    grid_html = render_grid_html(env, agent_type, percept_enabled, st.session_state.q_agent if agent_type == "Q-Learning" else None)
    st.markdown(grid_html, unsafe_allow_html=True)

# Training Progress Visualization
if agent_type == "Q-Learning" and st.session_state.training_history:
    st.write("### 📈 Lern-Fortschritt (Steps pro Episode)")
    st.line_chart(st.session_state.training_history)
    st.caption("Ziel: Sinkende Kurve (Konvergenz) -> Der Agent findet den Weg schneller.")

# Q-Value Heatmap (Subtitle)
if agent_type == "Q-Learning" and st.session_state.q_agent:
    st.write("### Wissens-Karte (Max Q-Wert)")
    q_grid = np.max(st.session_state.q_agent.q, axis=2)
    st.dataframe(pd.DataFrame(q_grid).style.background_gradient(cmap="Greens", axis=None))

# --- 7. LOGS ---
st.markdown("---")
with st.expander("🧠 Agenten Gedanken-Protokoll (Live Logik)", expanded=True):
    if st.session_state.logs:
        for log in reversed(st.session_state.logs[-10:]):
            st.markdown(f"- {log}", unsafe_allow_html=True)
    else:
        st.write("Noch keine Gedanken. Warte auf Exploration...")
