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

    def get_fog_view(self, radius=1):
        view = set()
        r, c = self.agent_pos
        for i in range(-radius, radius+1):
            for j in range(-radius, radius+1):
                if abs(i) + abs(j) <= radius: # Strict Manhattan distance (only adjacent if radius=1)
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
            st.session_state.logs.append(f"🎲 **Zufalls-Zug** (Exploration) bei {s}")
            return random.randint(0, 3)
        
        # Grease the wheels: explain greedy
        vals = self.q[s[0], s[1]]
        max_v = np.max(vals)
        action = np.argmax(vals)
        st.session_state.logs.append(f"🧠 **Gieriger Zug** (Exploitation) bei {s} (Q={max_v:.2f})")
        return action

    def learn(self, s, r, s_next):
        if self.prev_s is None: return
        
        old_q = self.q[self.prev_s[0], self.prev_s[1], self.prev_a]
        next_max = np.max(self.q[s_next[0], s_next[1]])
        
        new_q = old_q + self.alpha * (r + self.gamma * next_max - old_q)
        self.q[self.prev_s[0], self.prev_s[1], self.prev_a] = new_q
        
        # Didactic Log
        actions = ["OBEN", "UNTEN", "LINKS", "RECHTS"]
        act_str = actions[self.prev_a]
        log_msg = (f"🎯 **Q-Update** @ {self.prev_s} Aktion {act_str}:<br>"
                   f"`Q_neu = {old_q:.2f} + {self.alpha} * ({r:.2f} + {self.gamma} * {next_max:.2f} - {old_q:.2f})` "
                   f"➡️ **{new_q:.3f}**")
        st.session_state.logs.append(log_msg)
        
    def post_step(self, s, a):
        self.prev_s = s
        self.prev_a = a

# --- 3. STATE INITIALIZATION ---
    st.session_state.env = Environment(10, 10, step_penalty=-0.1, goal_reward=100.0)
    st.session_state.agent_str = "Manuell"
    st.session_state.logs = []
    st.session_state.q_agent = None
    
    # Performance Stats (Per Agent Type)
    # Structure: {AgentName: {'wins': 0, 'episodes': 0, 'total_steps': 0, 'total_return': 0.0}}
    st.session_state.stats = {
        "Manuell": {'wins': 0, 'episodes': 0, 'total_steps': 0, 'total_return': 0.0},
        "Reflex-Agent": {'wins': 0, 'episodes': 0, 'total_steps': 0, 'total_return': 0.0},
        "Modell-basiert": {'wins': 0, 'episodes': 0, 'total_steps': 0, 'total_return': 0.0},
        "Q-Learning": {'wins': 0, 'episodes': 0, 'total_steps': 0, 'total_return': 0.0}
    }
    # Current Episode Tracker
    st.session_state.current_episode = {'steps': 0, 'return': 0.0}

# --- 4. SIDEBAR ---
st.sidebar.title("LABOR STEUERUNG")

# Grid Resizer
grid_n = st.sidebar.slider("Gittergröße N", 5, 20, 10)
if grid_n != st.session_state.env.width:
    st.session_state.env = Environment(grid_n, grid_n)
    st.session_state.q_agent = None # Reset Q

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

st.sidebar.markdown("---")
st.sidebar.subheader("Simulations-Steuerung")
auto_run = st.sidebar.checkbox("Auto-Lauf (Simulation)", value=False)
speed = st.sidebar.slider("Geschwindigkeit (Wartezeit in s)", 0.0, 1.0, 0.2)
fog_enabled = st.sidebar.checkbox("Nebel des Krieges (Fog of War)", value=True, help="Wenn aktiv, sieht der Agent nur benachbarte Felder.")

st.sidebar.markdown("---")
st.sidebar.subheader("Umgebungs-Konfiguration")
env_step_penalty = st.sidebar.slider("Schritt-Strafe (Kosten)", -2.0, 0.0, -0.1, 0.1, help="Kosten (negativ) für jeden Schritt.")
env_goal_reward = st.sidebar.slider("Ziel-Belohnung", 10.0, 200.0, 100.0, 10.0, help="Belohnung für das Erreichen des Ziels.")

if st.sidebar.button("RESET SIMULATION"):
    st.session_state.env = Environment(grid_n, grid_n, step_penalty=env_step_penalty, goal_reward=env_goal_reward)
    st.session_state.q_agent = None
    st.session_state.logs = []
    st.session_state.current_episode = {'steps': 0, 'return': 0.0}
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

def render_grid_html(env, agent_type, fog_enabled, q_agent=None):
    visible_mask = env.get_fog_view(radius=1) if fog_enabled else {(r,c) for r in range(env.height) for c in range(env.width)}
    
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
            symbol = "░" # Fog default
            
            # Q-Learning Visualization (Heatmap)
            # We want to show Q-values even in Fog for the 'Mind' of the agent? 
            # Or only observed? Usually Q-values are internal state, so we can show them always or dependent on fog?
            # Let's show them 'underneath' the fog if we want to debug, but strictly speaking 
            # purely visual Fog should hide everything. 
            # BUT: The requirement is "learning should be visible". 
            # So we will tint the cell based on Q-value if visited/known.
            
            q_color = None
            if q_agent:
                # Get max Q for this state
                q_vals = q_agent.q[r, c]
                best_q = np.max(q_vals)
                if best_q != 0:
                    # Simple Green intensity for positive Q
                    intensity = int(min(255, max(50, (best_q / max_q_val) * 200))) 
                    # If negative (hitting walls), maybe Red?
                    if best_q < 0:
                         q_color = f"rgba(255, 0, 0, 0.3)"
                    else:
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
                # In Fog
                # Check Memory for Model-Based
                if agent_type == "Modell-basiert" and pos in env.memory_map:
                    val = env.memory_map[pos]
                    if val == 1: symbol = "▒" # Ghost Wall
                    elif val == 2: symbol = "⚐" # Ghost Goal
                    else: symbol = "&nbsp;" # Empty Known
                    style = "color: #555;" # Dimmed for memory
                
                # Check Q-Values for Q-Learning (Internal Knowledge is 'clear' to the agent)
                # If we want to visualize what the agent KNOWS, we should probably show the color even in fog?
                # Let's show the Q-color in fog but with existing Fog char '░' or '·'?
                # Better: IF Q-value is non-zero, it means agent has explored there.
                # So we show the Q-color.
                elif q_agent and np.max(q_agent.q[r, c]) != 0:
                     symbol = "·"
                     if q_color: style = f"background-color: {q_color}; color: #888;"

            # Construct Cell HTML
            # We need a span to apply style
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
    st.write("Du bist der Agent. Nutze die Pfeiltasten, um durch den Nebel zu navigieren. Beobachte, wie schwierig es ist, Entscheidungen ohne vollständige Information zu treffen. Dies simuliert die **Partielle Beobachtbarkeit**.")
elif agent_type == "Reflex-Agent":
    st.markdown('<div class="theory-title">MODUS: REFLEX-AGENT (Einfach)</div>', unsafe_allow_html=True)
    st.image(r"https://latex.codecogs.com/png.latex?\color{green}\text{Aktion}(p) = \text{Regel}[\text{Sensor}(p)]")
    st.write("Dieser Agent **reagiert** nur auf das aktuelle Feld. Er hat KEIN Gedächtnis. Wenn er vor einer Wand steht, dreht er sich um. Er verfängt sich oft in Endlos-Schleifen, da er nicht weiß, wo er schon war.")
elif agent_type == "Modell-basiert":
    st.markdown('<div class="theory-title">MODUS: MODELL-BASIERTER REFLEX-AGENT</div>', unsafe_allow_html=True)
    st.write("Dieser Agent besitzt ein **Internes Modell** ($S'$). Er merkt sich, wo er schon war (Mental Map) und 'lichtet' den Nebel in seinem Gedächtnis. Das erlaubt ihm, effizienter zu suchen und Sackgassen zu vermeiden.")
elif agent_type == "Q-Learning":
    st.markdown('<div class="theory-title">MODUS: Q-LEARNING (Verstärkendes Lernen)</div>', unsafe_allow_html=True)
    st.latex(r"Q(s,a) \leftarrow Q(s,a) + \alpha [R + \gamma \max_{a'} Q(s',a') - Q(s,a)]")
    st.write(f"Der Agent lernt den **Nutzen (Utility)** von Aktionen durch Belohnung ($R$) und Bestrafung. Er baut eine Tabelle (Q-Table) auf, die ihm sagt: 'In Zustand $s$, wie gut ist Aktion $a$?'.")
    st.write(f"- $\\alpha={alpha}$: **Lernrate**. Wie schnell überschreibt neues Wissen das alte?")
    st.write(f"- $\\gamma={gamma}$: **Diskount-Faktor**. Wie wichtig sind zukünftige Belohnungen?")
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

st.write("Steuerung: Nutze **Pfeiltasten** oder Buttons.")

# Live Metrics
curr_ep = st.session_state.current_episode
col_m1, col_m2 = st.columns(2)
col_m1.metric("Schritte (Aktuell)", curr_ep['steps'])
col_m2.metric("Return (Aktuell)", f"{curr_ep['return']:.2f}")

action = None # Initialize action to avoid NameError

# MANUAL INPUT MAPPING
if agent_type == "Manuell":
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("OBEN ⬆️"): action = 0
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("LINKS ⬅️"): action = 2
    with col2:
        if st.button("UNTEN ⬇️"): action = 1
    with col3:
        if st.button("RECHTS ➡️"): action = 3

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
                # Dumb rule: Random safe move, bias towards goal if visible
                view = env.get_fog_view(1)
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
            elif agent_type == "Modell-basiert":
                # Update memory
                c_view = env.get_fog_view(1)
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

            # EXECUTE ACTION inside the loop (only for auto-run agents)
            if action is not None and not env.game_over:
                next_s, r, done = env.step(action)
                
                # Update Stats
                st.session_state.current_episode['steps'] += 1
                st.session_state.current_episode['return'] += r
                
                # Post-Step Learning
                if agent_type == "Q-Learning" and st.session_state.q_agent:
                    st.session_state.q_agent.learn(None, r, next_s) # s is stored in prev_s
                
                # Render Loop for Auto-Run
                if steps_to_run > 1:
                     # Re-render Grid using helper
                     grid_html = render_grid_html(env, agent_type, fog_enabled, st.session_state.q_agent if agent_type == "Q-Learning" else None)
                     placeholder.markdown(grid_html, unsafe_allow_html=True)
                     
                     # Update Live Metrics in Loop (requires placeholder or rerun, but rerun breaks loop)
                     # We can't easily update the 'st.metric' outside without rerun. 
                     # But we can assume the user sees the final result or we use another placeholder.
                     # For simplicity, we just run.
                     
                     time.sleep(speed)
                     if done: 
                         st.balloons()
                         st.session_state.logs.append("ZIEL ERREICHT!")
                         
                         # Save Session Stats
                         stat_entry = st.session_state.stats[agent_type]
                         stat_entry['episodes'] += 1
                         stat_entry['wins'] += 1 # Only goal triggers done=True with reward 100 in this env usually
                         stat_entry['total_steps'] += st.session_state.current_episode['steps']
                         stat_entry['total_return'] += st.session_state.current_episode['return']
                         
                         # Reset Current
                         st.session_state.current_episode = {'steps': 0, 'return': 0.0}
                         
                         break

        # If auto-run is on and game not over, rerun to continue loop
        if auto_run and not env.game_over:
             time.sleep(speed)
             st.rerun()

# MANUAL EXECUTION
if agent_type == "Manuell" and action is not None and not env.game_over:
    next_s, r, done = env.step(action)
    
    # Update Stats
    st.session_state.current_episode['steps'] += 1
    st.session_state.current_episode['return'] += r
    
    if done:
        st.balloons()
        st.session_state.logs.append("ZIEL ERREICHT!")
        
        # Save Session Stats
        stat_entry = st.session_state.stats[agent_type]
        stat_entry['episodes'] += 1
        stat_entry['wins'] += 1
        stat_entry['total_steps'] += st.session_state.current_episode['steps']
        stat_entry['total_return'] += st.session_state.current_episode['return']
        
        # Reset Current
        st.session_state.current_episode = {'steps': 0, 'return': 0.0}
        
    # Removed st.rerun() to allow balloons to render and grid to update in current frame

# --- 6. RENDERER (ASCII/EMOJI) ---

# Only render the static grid if NOT in auto-run loop (to avoid duplicate rendering or flashing)
if not (agent_type != "Manuell" and auto_run):
    grid_html = render_grid_html(env, agent_type, fog_enabled, st.session_state.q_agent if agent_type == "Q-Learning" else None)
    st.markdown(grid_html, unsafe_allow_html=True)

# Q-Value Heatmap (Subtitle)
if agent_type == "Q-Learning" and st.session_state.q_agent:
    st.write("### Wissens-Karte (Max Q-Wert)")
    # Render small table/grid
    q_grid = np.max(st.session_state.q_agent.q, axis=2)
    st.dataframe(pd.DataFrame(q_grid).style.background_gradient(cmap="Greens", axis=None))

# --- 7. AGENT THOUGHTS (LOGS) ---
st.markdown("---")
with st.expander("🧠 Agenten Gedanken-Protokoll (Live Logik)", expanded=True):
    # Show last 5 logs reversed
    if st.session_state.logs:
        for log in reversed(st.session_state.logs[-10:]):
            st.markdown(f"- {log}", unsafe_allow_html=True)
    else:
        st.write("Noch keine Gedanken. Warte auf Exploration...")
