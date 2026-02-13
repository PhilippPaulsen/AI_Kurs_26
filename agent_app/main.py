import streamlit as st
import time
import pandas as pd
import numpy as np
import random
from environment import Grid
from agents import ManualAgent, SimpleReflexAgent, ModelBasedReflexAgent, QLearningAgent

# Configure Page
st.set_page_config(page_title="AI Agent Playground", layout="wide")

# Custom CSS for Monospace Grid
st.markdown("""
<style>
.grid-container {
    font-family: 'Courier New', Courier, monospace;
    white-space: pre;
    font-size: 20px;
    line-height: 20px;
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    display: inline-block;
}
.stButton>button {
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'env' not in st.session_state:
    st.session_state.env = Grid(10, 10)
    st.session_state.agent = ManualAgent()
    st.session_state.agent_type = "Manual"
    st.session_state.step_count = 0
    st.session_state.total_reward = 0
    st.session_state.game_over = False
    st.session_state.auto_run = False
    st.session_state.history = []

# Sidebar Configuration
st.sidebar.header("Configuration")

# Grid Settings
grid_size = st.sidebar.slider("Grid Size (NxN)", 5, 20, 10)
wall_coverage = st.sidebar.slider("Wall Coverage", 0.0, 0.4, 0.2)
reset_grid = st.sidebar.button("Reset Environment")

# Agent Settings
agent_type = st.sidebar.selectbox(
    "Agent Type",
    ["Manual", "Simple Reflex", "Model-based Reflex", "Q-Learning"]
)

# Q-Learning Hyperparameters
if agent_type == "Q-Learning":
    st.sidebar.subheader("Q-Learning Parameters")
    alpha = st.sidebar.slider("Learning Rate (α)", 0.1, 1.0, 0.5)
    gamma = st.sidebar.slider("Discount Factor (γ)", 0.1, 1.0, 0.9)
    epsilon = st.sidebar.slider("Exploration Rate (ε)", 0.0, 1.0, 0.1)

# Speed Control
speed = st.sidebar.slider("Simulation Speed (sec/step)", 0.0, 2.0, 0.5)

# Reset Logic
if reset_grid or agent_type != st.session_state.agent_type or (grid_size != st.session_state.env.width):
    st.session_state.env = Grid(grid_size, grid_size, wall_coverage)
    st.session_state.agent_type = agent_type
    st.session_state.step_count = 0
    st.session_state.total_reward = 0
    st.session_state.game_over = False
    st.session_state.history = []
    
    if agent_type == "Manual":
        st.session_state.agent = ManualAgent()
    elif agent_type == "Simple Reflex":
        st.session_state.agent = SimpleReflexAgent()
    elif agent_type == "Model-based Reflex":
        st.session_state.agent = ModelBasedReflexAgent()
    elif agent_type == "Q-Learning":
        # Preserve Q-table if re-selecting Q-Learning on same grid size?
        # For simplicity, reset agent to allow parameter updates or fresh start
        st.session_state.agent = QLearningAgent(alpha, gamma, epsilon)

    st.rerun()

# --- Main Logic ---

env = st.session_state.env
agent = st.session_state.agent

def step_agent(action_intent=None):
    if st.session_state.game_over:
        return

    # 1. Perceive
    current_state = env.agent_pos # (row, col)
    
    # 2. Decide Action
    if isinstance(agent, ManualAgent):
        action = agent.act(None, action_intent)
        if action is None: return # Wait for input
    elif isinstance(agent, QLearningAgent):
        action = agent.act(current_state)
    else: # Reflex Agents
        percept = env.get_percept()
        action = agent.act(percept)
    
    # 3. Execute Action
    next_state, reward, done = env.step(action)
    
    # 4. Learn (if applicable)
    if isinstance(agent, QLearningAgent):
        agent.learn(next_state, reward) # Note: learn uses prev_state stored in agent
    elif isinstance(agent, ModelBasedReflexAgent):
        agent.update_internal_model(next_state)

    # 5. Update State
    st.session_state.step_count += 1
    st.session_state.total_reward += reward
    st.session_state.game_over = done
    
    # Log
    if hasattr(agent, 'log') and agent.log:
        st.session_state.history.append(f"Step {st.session_state.step_count}: {agent.log[-1]}")

# UI Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Grid World")
    
    # Render ASCII Grid
    # Create visual representation
    grid_display = ""
    rows = []
    for r in range(env.height):
        cols = []
        for c in range(env.width):
            char = "."
            if (r, c) == env.agent_pos:
                char = "🤖" # A
            elif (r, c) == env.goal_pos:
                char = "🏁" # G
            elif env.grid[r, c] == 1:
                char = "🧱" # #
            elif (r, c) in env.visited:
                char = "👣"
            
            cols.append(char)
        rows.append(" ".join(cols))
    grid_display = "\n".join(rows)

    st.text(grid_display) # Using text ensure monospaced alignment if simple string
    # Try st.code for better monospace block or markdown with pre
    st.markdown(f'<div class="grid-container">{grid_display}</div>', unsafe_allow_html=True)
    
    if st.session_state.game_over:
        st.success(f"Goal Reached! Total Reward: {st.session_state.total_reward:.1f}")
        if st.button("Play Again"):
            env.reset()
            st.session_state.game_over = False
            st.session_state.step_count = 0
            st.session_state.total_reward = 0
            st.session_state.history = []
            if hasattr(agent, 'reset'): agent.reset()
            st.rerun()

    # Manual Controls
    if agent_type == "Manual" and not st.session_state.game_over:
        c1, c2, c3 = st.columns([1,1,1])
        with c2:
            if st.button("⬆️"): step_agent(0); st.rerun()
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            if st.button("⬅️"): step_agent(2); st.rerun()
        with c2:
            if st.button("⬇️"): step_agent(1); st.rerun()
        with c3:
            if st.button("➡️"): step_agent(3); st.rerun()

with col2:
    st.subheader("Agent Internals")
    
    # Didactic Info
    if agent_type == "Simple Reflex":
        st.info("ℹ️ **Simple Reflex Agent**: Only sees immediate neighbors. Can get stuck in loops or local optima if the path requires moving away from the goal temporarily.")
    elif agent_type == "Model-based Reflex":
        st.info("ℹ️ **Model-based Agent**: Remembers where it has been. Better at exploration but still lacks a global plan.")
    elif agent_type == "Q-Learning":
        st.info("ℹ️ **Q-Learning Agent**: Learns the value of actions over time. Initially random, it converges to the optimal path.")

    # Controls
    if agent_type != "Manual":
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Step"):
                step_agent()
                st.rerun()
        with c2:
            auto = st.checkbox("Auto Run", key="auto_run_check")
            if auto:
                step_agent()
                time.sleep(speed)
                st.rerun()

    # Stats
    st.metric("Steps", st.session_state.step_count)
    st.metric("Reward", f"{st.session_state.total_reward:.1f}")

    # Logs
    st.subheader("Thought Process")
    log_container = st.container(height=200)
    for log in reversed(st.session_state.history[-10:]):
        log_container.text(log)

# Q-Table Visualization
if agent_type == "Q-Learning":
    st.subheader("Q-Table (Learned Values)")
    # Create DataFrame for Q-Table
    data = []
    for r in range(env.height):
        for c in range(env.width):
            state = (r, c)
            q_vals = agent.get_q(state)
            best_action = ["Up", "Down", "Left", "Right"][np.argmax(q_vals)]
            max_q = np.max(q_vals)
            
            # Only show interesting states (visited or walls/goals)
            if np.any(q_vals != 0):
                row = {
                    "Pos": str(state),
                    "Up": f"{q_vals[0]:.2f}",
                    "Down": f"{q_vals[1]:.2f}",
                    "Left": f"{q_vals[2]:.2f}",
                    "Right": f"{q_vals[3]:.2f}",
                    "Best": best_action
                }
                data.append(row)
    
    if data:
        st.dataframe(pd.DataFrame(data))
    else:
        st.text("No Q-values learned yet.")

