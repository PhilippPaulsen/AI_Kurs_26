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

html, body, .stApp {
    background-color: #000000;
    color: #ffffff;
    font-family: 'Roboto Mono', monospace;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0a0a0a;
    border-right: 1px solid #333;
}

/* Streamlit Widget Labels */
[data-testid="stWidgetLabel"] p {
    color: #ffffff !important;
    font-weight: 500 !important;
}

/* Metrics labels */
[data-testid="stMetricLabel"] {
    color: #e0e0e0 !important;
}

/* Captions and Help Text */
[data-testid="stCaptionContainer"], .stCaption {
    color: #cccccc !important;
}

/* Metrics values */
[data-testid="stMetricValue"] {
    color: #ffffff !important;
}

/* Sidebar Titles and Subheaders */
[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3 {
    color: #00ff00 !important;
    text-shadow: 0 0 5px rgba(0, 255, 0, 0.4);
}

/* Sidebar Section Dividers/Labels */
.st-emotion-cache-1vt4y6f {
    color: #00ff00 !important;
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
    def __init__(self, width=10, height=10, step_penalty=-0.1, goal_reward=100.0, wall_penalty=-5.0):
        self.width = width
        self.height = height
        self.step_penalty = step_penalty
        self.goal_reward = goal_reward
        self.wall_penalty = wall_penalty
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

    def reset_agent(self):
        # Reset only agent state, keep environment (walls)
        self.agent_pos = self.start_pos
        self.visited = {self.start_pos}
        self.game_over = False
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
                # Update memory if bumped
                self.memory_map[(nr, nc)] = 1 
                return self.agent_pos, self.wall_penalty, False
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
    st.session_state.env = Environment(10, 10, step_penalty=-0.1, goal_reward=100.0, wall_penalty=-5.0)
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
    st.session_state.current_episode = {'steps': 0, 'return': 0.0, 'last_reward': 0.0, 'last_action': None}
    # Training History for Q-Learning
    st.session_state.training_history = []

# --- 4. SIDEBAR ---


# Grid Resizer
grid_n = st.sidebar.slider("Gittergröße N", 5, 20, 10)
if grid_n != st.session_state.env.width:
    st.session_state.env = Environment(grid_n, grid_n)
    st.session_state.q_agent = None # Reset Q
    st.session_state.training_history = []

# Agent Select
agent_type = st.sidebar.selectbox("Agenten Intelligenz", ["Manuell", "Reflex-Agent", "Modell-basiert", "Q-Learning"])
st.session_state.agent_str = agent_type

# Progression Display
st.sidebar.caption("Entwicklungsstufe:")
if agent_type == "Reflex-Agent":
    st.sidebar.markdown("**Reactive** → Model-Based → Learning")
elif agent_type == "Modell-basiert":
    st.sidebar.markdown("Reactive → **Model-Based** → Learning")
elif agent_type == "Q-Learning":
    st.sidebar.markdown("Reactive → Model-Based → **Learning**")
else: # Manual
    st.sidebar.markdown("<span style='color:#777'>Reactive → Model-Based → Learning</span>", unsafe_allow_html=True)

# Q-Params (Context Sensitive)
alpha, gamma, epsilon = 0.5, 0.9, 0.1
if agent_type == "Q-Learning":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Hyperparameter (Lernen)")
    alpha = st.sidebar.slider("Alpha (Lernrate)", 0.1, 1.0, 0.5, help="Lernrate (0.0 - 1.0). Bestimmt, wie stark neue Informationen alte überschreiben (0.5 = balanciert).")
    gamma = st.sidebar.slider("Gamma (Diskount)", 0.1, 1.0, 0.9, help="Diskount-Faktor (0.0 - 1.0). Wichtigkeit zukünftiger Belohnungen. Nahe 1 (z.B. 0.9) fördert langfristiges Planen.")
    epsilon = st.sidebar.slider("Epsilon (Exploration)", 0.0, 1.0, 0.1, help="Explorations-Rate (0.0 - 1.0). Wahrscheinlichkeit für zufällige Züge, um Neues zu entdecken und lokalen Optima zu entkommen.")
    
    train_episodes = st.sidebar.slider("Training-Episoden", 1, 500, 50)
    if st.sidebar.button(f"Train Episodes ({train_episodes}x)"):
        # Training Loop
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        # Ensure Agent Exists
        if st.session_state.q_agent is None:
             st.session_state.q_agent = QAgent(st.session_state.env, alpha, gamma, epsilon)
        
        qa = st.session_state.q_agent
        # Update params
        qa.alpha, qa.gamma, qa.epsilon = alpha, gamma, epsilon
        
        recent_rewards = []

        for ep in range(train_episodes):
            # Soft Reset (Keep Walls)
            env = st.session_state.env
            env.agent_pos = env.start_pos
            env.game_over = False
            env.visited = {env.start_pos}
            
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
            st.session_state.training_history.append({'episode': ep+1, 'steps': steps, 'reward': ep_return})
            recent_rewards.append(ep_return)
            if len(recent_rewards) > 10: recent_rewards.pop(0)
            
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            
            progress_bar.progress((ep + 1) / train_episodes)
            status_text.text(f"Ep {ep+1}/{train_episodes} | Ø Reward (10): {avg_reward:.1f}")
        
        st.session_state.logs.append(f"✅ **Training abgeschlossen:** {train_episodes} Episoden.")
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("Simulations-Steuerung")
auto_run = st.sidebar.checkbox("Auto-Lauf (Simulation)", value=False)
speed = st.sidebar.slider("Geschwindigkeit (Wartezeit in s)", 0.0, 1.0, 0.2)
percept_enabled = st.sidebar.checkbox("Percept Field (Sichtfeld)", value=True, help="Wenn aktiv, sieht der Agent nur benachbarte Felder (Radius 1).")

# NEW: Current Percept Expander
with st.sidebar.expander("Perception (Current Observation)", expanded=True):
    percepts = st.session_state.env.get_current_percept_text()
    
    # Clean CSS Grid Layout for styling
    p_up, p_down = percepts['UP'], percepts['DOWN']
    p_left, p_right = percepts['LEFT'], percepts['RIGHT']
    
    st.markdown(f"""
    <div style="display: grid; grid-template-columns: 40px 1fr; gap: 2px; align-items: center; font-family: monospace; font-size: 0.9em;">
        <div style="text-align: right;">⬆️</div><div style="color: #eee;">{p_up}</div>
        <div style="text-align: right;">⬅️</div><div style="color: #eee;">{p_left}</div>
        <div style="text-align: right;">➡️</div><div style="color: #eee;">{p_right}</div>
        <div style="text-align: right;">⬇️</div><div style="color: #eee;">{p_down}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Compact Footnote
    st.caption("Radius 1. 'EMPTY' = leer.")

st.sidebar.markdown("---")
st.sidebar.subheader("Umgebungs-Konfiguration")
env_step_penalty = st.sidebar.slider("Schritt-Strafe (Kosten)", -2.0, 0.0, -0.1, 0.1, help="Kosten (negativ) für jeden Schritt.")
env_wall_penalty = st.sidebar.slider("Wand-Strafe (Kollision)", -10.0, -1.0, -5.0, 1.0, help="Strafe (negativ) für das Laufen gegen eine Wand.")
env_goal_reward = st.sidebar.slider("Ziel-Belohnung", 10.0, 200.0, 100.0, 10.0, help="Belohnung für das Erreichen des Ziels.")
# Update active environment
st.session_state.env.step_penalty = env_step_penalty
st.session_state.env.wall_penalty = env_wall_penalty
st.session_state.env.goal_reward = env_goal_reward



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
                style = "color: #777;" # Brighter Shadow
                
                # Check Memory for Model-Based
                if agent_type == "Modell-basiert" and pos in env.memory_map:
                    val = env.memory_map[pos]
                    if val == 1: symbol = "▒" # Ghost Wall
                    elif val == 2: symbol = "⚐" # Ghost Goal
                    else: symbol = "&nbsp;" # Empty Known
                    style = "color: #aaa;" # Dimmed for memory
                
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

# --- 5. MAIN LOOP & UI LAYOUT ---

env = st.session_state.env

# 1. DEFINE ZONES (Visual Hierarchy)
# Zone A: Environment (Top, visual context)
zone_env = st.container()

# Zone B: Agent Status (Compact, immediate feedback)
zone_agent = st.container()

# Zone C: Didactics (Collapsible, reflection)
zone_didactics = st.expander("🎓 Didaktik: Warum verhält sich der Agent so?", expanded=False)

# Zone D: Actions (Controls)
zone_actions = st.container()


# 2. FILL ZONE C: DIDACTICS
with zone_didactics:
    # Custom CSS for Tags
    st.markdown("""
    <style>
    .agent-tag {
        background-color: #333;
        color: #ddd;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        margin-right: 5px;
        border: 1px solid #555;
    }
    </style>
    """, unsafe_allow_html=True)

    # Agent-Specific Content
    if agent_type == "Manuell":
        st.write("Policy wird vollständig vom Menschen bestimmt.")
        st.markdown('<span class="agent-tag">Policy: human</span> <span class="agent-tag">Memory: none</span> <span class="agent-tag">Planning: none</span>', unsafe_allow_html=True)
        st.write("---")
        st.markdown("**🎯 Lernfokus:** Wie beeinflusst deine eigene Strategy den Return?")
        with st.expander("🤔 Reflexionsfragen"):
            st.markdown("- Welche Information nutzt du zur Entscheidung?\n- Wie würdest du deine Policy beschreiben?\n- Reagierst du eher lokal oder planst du voraus?")
        
        with st.expander("📐 Theorie (Optional)"):
            st.write("<b>Policy wird vollständig extern (vom Menschen) bestimmt.</b>", unsafe_allow_html=True)
            st.markdown("- Agent = Perception → Action\n- Kein automatisches Lernen\n- Performance hängt von Strategy ab")
            st.markdown(r"Notation: $\pi(a|p)$")
            st.caption("Transferfrage: Wie würdest du deine eigene Policy formalisieren?")

    elif agent_type == "Reflex-Agent":
        st.write("Action basiert nur auf aktuellem Percept (keine Memory).")
        st.markdown('<span class="agent-tag">Policy: reactive</span> <span class="agent-tag">Memory: none</span> <span class="agent-tag">Planning: none</span>', unsafe_allow_html=True)
        st.write("---")
        st.markdown("**🎯 Lernfokus:** Was passiert bei Partial Observability ohne Gedächtnis?")
        with st.expander("🤔 Reflexionsfragen"):
            st.markdown("- Warum wiederholt der Agent möglicherweise ineffiziente Bewegungen?\n- Welche Information fehlt ihm?\n- Würde ein internes State-Modell helfen?")
        
        with st.expander("📐 Theorie (Optional)"):
            st.write("<b>Action basiert ausschließlich auf aktuellem Percept (kein Memory).</b>", unsafe_allow_html=True)
            st.markdown("- Markov-Annahme\n- Keine interne State-Erweiterung\n- Problematisch bei Partial Observability")
            st.markdown(r"Notation: $a = \pi(p)$")
            st.caption("Transferfrage: Warum kann der Agent ineffiziente Zyklen wiederholen?")

    elif agent_type == "Modell-basiert":
        st.write("Interner State speichert vergangene Information.")
        st.markdown('<span class="agent-tag">Policy: reactive</span> <span class="agent-tag">Memory: internal map</span> <span class="agent-tag">Planning: limited</span>', unsafe_allow_html=True)
        st.write("---")
        st.markdown("**🎯 Lernfokus:** Wie kompensiert internes Gedächtnis fehlende Observation?")
        with st.expander("🤔 Reflexionsfragen"):
            st.markdown("- Welche Information speichert der Agent?\n- Wann ist Model-Based besser als Reflex?\n- Ist der Agent jetzt optimal oder nur informierter?")
            
        with st.expander("📐 Theorie (Optional)"):
            st.write("<b>Interner State erweitert die Information über das aktuelle Percept hinaus.</b>", unsafe_allow_html=True)
            st.markdown("- State ≠ Perception\n- Gedächtnis kompensiert Informationslücken\n- Besser bei Partial Observability")
            st.latex(r"State_t = f(State_{t-1}, Percept_t)")
            st.caption("Transferfrage: Welche Information speichert der Agent zusätzlich?")

    elif agent_type == "Q-Learning":
        st.write("Policy wird durch Reward-Lernen angepasst.")
        st.markdown('<span class="agent-tag">Policy: learned</span> <span class="agent-tag">Memory: Q-table</span> <span class="agent-tag">Exploration: ε-greedy</span>', unsafe_allow_html=True)
        st.write("---")
        st.markdown("**🎯 Lernfokus:** Exploration vs. Exploitation.")
        with st.expander("🤔 Reflexionsfragen"):
            st.markdown("- Wie beeinflusst ε das Verhalten?\n- Warum steigt der Return mit Training?\n- Was bedeutet Konvergenz?")

        with st.expander("📐 Theorie (Optional)"):
            st.write("<b>Policy wird durch iteratives Reward-Lernen optimiert.</b>", unsafe_allow_html=True)
            st.markdown("- Exploration vs. Exploitation\n- Value-basierte Entscheidung\n- Konvergenz bei ausreichend Training")
            st.latex(r"Q(s,a) \leftarrow r + \gamma \max Q(s',a')")
            st.caption("Transferfrage: Wie beeinflusst ε die Balance zwischen Exploration und Exploitation?")

    # Didactic box with live analysis
    analysis_text = ""
    if agent_type == "Manual":
        analysis_text = "Du entscheidest selbst. Deine Strategie bestimmt den Erfolg."
    elif agent_type == "Reflex-Agent":
        analysis_text = "Der Agent reagiert blind auf das aktuelle Feld. Ohne Gedächtnis tappt er in Fallen."
    elif agent_type == "Modell-basiert":
        analysis_text = "Die interne Karte wächst mit jedem Schritt. Der Agent plant aber nicht weit voraus."
    elif agent_type == "Q-Learning":
        analysis_text = f"Der Agent aktualisiert Q(s,a) basierend auf Reward ({st.session_state.env.step_penalty}) und Zukunftserwartung (γ={gamma})."
    
    st.info(f"**🔍 Live-Analyse:** {analysis_text}")

    if not auto_run:
        st.caption("Reflexionsfrage: Welche Information fehlt dir gerade, um optimal zu handeln?")


# 3. FILL ZONE A: ENVIRONMENT HEADER
with zone_env:
    st.caption("Environment: Grid World")
    if percept_enabled:
        st.caption("Observability: Partially Observable (radius = 1)")
    else:
        st.caption("Observability: Fully Observable (complete observation)")
    
    # Placeholder for the Grid (to be rendered later)
    grid_placeholder = st.empty()


# 4. FILL ZONE B: AGENT METRICS
with zone_agent:
    curr_ep = st.session_state.current_episode
    
    # Primary Key Metric: Return
    c_main, c_details = st.columns([1, 3])
    c_main.metric("Return (Gₜ)", f"{curr_ep['return']:.2f}", help="Gₜ = Summe aller Rewards")
    
    # Compact Details Bar
    last_act = curr_ep.get('last_action', '-')
    state_str = str(st.session_state.env.agent_pos)
    last_rew = f"{curr_ep.get('last_reward', 0.0):.2f}"
    steps_str = f"{curr_ep['steps']} / 20"
    
    with c_details:
        st.markdown(f"""
        <div style="background-color: #111; padding: 10px; border-radius: 5px; border: 1px solid #333; margin-top: 5px;">
            <span style="color: #bbb; margin-right: 15px;">Episode Steps: <strong style="color: #fff;">{steps_str}</strong></span>
            <span style="color: #bbb; margin-right: 15px;">State: <strong style="color: #fff;">{state_str}</strong></span>
            <span style="color: #bbb; margin-right: 15px;">rₜ: <strong style="color: #fff;">{last_rew}</strong></span>
            <span style="color: #bbb;">Last Action: <strong style="color: #fff;">{last_act}</strong></span>
        </div>
        """, unsafe_allow_html=True)
    
        if auto_run:
            st.caption("Auto mode running — open ‘Lernhinweise’ for analysis.")
        elif curr_ep['steps'] >= 20:
            st.info("🏁 **Phase Complete:** War dieser Agent effizienter als der vorherige? Warum?")

    # --- RESET BUTTON BAR (Zone B Bottom) ---
    st.markdown("---")
    # State for confirmations
    if 'confirm_reset' not in st.session_state: st.session_state.confirm_reset = False
    if 'confirm_new_maze' not in st.session_state: st.session_state.confirm_new_maze = False

    b_col1, b_col2, b_col3 = st.columns(3)
    
    # 1. Reset Episode
    if b_col1.button("🔄 Reset Episode", help="Setzt Agenten auf Start zurück. Q-Table bleibt."):
        env.reset_agent()
        st.session_state.current_episode = {'steps': 0, 'return': 0.0, 'last_reward': 0.0, 'last_action': None}
        st.rerun()

    # 2. Brain Reset (Confirm)
    with b_col2:
        if not st.session_state.confirm_reset:
            if st.button("🧠 Brain Reset"):
                st.session_state.confirm_reset = True
                st.rerun()
        else:
            if st.button("⚠️ Wirklich löschen?", type="primary"):
                st.session_state.q_agent = None
                st.session_state.training_history = []
                st.session_state.current_episode = {'steps': 0, 'return': 0.0, 'last_reward': 0.0, 'last_action': None}
                st.session_state.confirm_reset = False
                st.rerun()
            st.caption("Klicke zum Bestätigen.")

    # 3. New Maze (Confirm)
    with b_col3:
        if not st.session_state.confirm_new_maze:
            if st.button("🎲 Neues Labyrinth"):
                st.session_state.confirm_new_maze = True
                st.rerun()
        else:
            if st.button("⚠️ Wirklich neu?", type="primary"):
                st.session_state.env = Environment(10, 10, step_penalty=env_step_penalty, goal_reward=env_goal_reward, wall_penalty=env_wall_penalty) # Re-init
                # Reset logic
                env = st.session_state.env # Update ref
                st.session_state.q_agent = None
                st.session_state.training_history = []
                st.session_state.current_episode = {'steps': 0, 'return': 0.0, 'last_reward': 0.0, 'last_action': None}
                st.session_state.confirm_new_maze = False
                st.rerun()
            st.caption("Klicke zum Bestätigen.")


# 5. INPUT LOGIC & ACTIONS (Zone D)
action = None # Initialize action

# Keyboard Logic (Hidden)
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


# MANUAL CONTROLS in Zone D
if agent_type == "Manuell":
    with zone_actions:
        st.markdown("### Actions")
        st.caption("Steuerung: Nutze **Pfeiltasten** oder Buttons.")
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
        # Use the placeholder in Zone A
        placeholder = grid_placeholder
        
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
                st.session_state.current_episode['last_reward'] = r
                st.session_state.current_episode['last_action'] = ["UP", "DOWN", "LEFT", "RIGHT"][action]                
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
    st.session_state.current_episode['last_reward'] = r
    st.session_state.current_episode['last_action'] = ["UP", "DOWN", "LEFT", "RIGHT"][action]    
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
    grid_placeholder.markdown(grid_html, unsafe_allow_html=True)

# Training Progress Visualization
if agent_type == "Q-Learning" and st.session_state.training_history:
    st.write("### 📈 Lern-Fortschritt (Reward pro Episode)")
    
    # Create DataFrame from history
    df_history = pd.DataFrame(st.session_state.training_history)
    
    # Check if 'reward' column exists (migration support)
    if 'reward' not in df_history.columns:
        # Fallback or empty if old format
        st.warning("Altes Datenformat erkannt. Bitte 'Brain Reset' durchführen.")
    else:
        st.line_chart(df_history.set_index("episode")['reward'])
    st.caption("Ziel: Steigende Kurve -> Der Agent maximiert seine Belohnung.")

# --- LIVE ANALYSIS DIDACTIC BOX ---
# Moved to Zone C (Didactics) at the top.


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
