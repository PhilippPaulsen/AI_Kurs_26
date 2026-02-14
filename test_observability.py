
import sys
import os
import random
import numpy as np

# Adjust path to import agent_app modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Mock Streamlit session state
class MockSessionState(dict):
    def __getattr__(self, key): return self.get(key)
    def __setattr__(self, key, val): self[key] = val

import streamlit as st
if not hasattr(st, 'session_state'):
    st.session_state = MockSessionState()

# Import Environment and QAgent classes by execing the file (since they are not in a module)
# This is a bit hacky but works for single-file apps without restructuring
# Better: Copy the class definitions or assume they are importable if main.py was a module.
# Since main.py is a script, let's just copy the relevant class definitions for testing or use a regex to extract them?
# Actually, I can just import them if I rename main.py to something else or if it doesn't run top-level code on import.
# main.py HAS top-level code (st.set_page_config etc).
# So I will copy the class definitions here for testing purposes to ensure logic correctness.

# ... Or I can just read the file and exec only the class definitions.
with open('agent_app/main.py', 'r') as f:
    content = f.read()

# Extract Class definitions (Environment and QAgent)
import ast
tree = ast.parse(content)
classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]

# Execute class definitions in local scope
for node in classes:
    exec(compile(ast.Module(body=[node], type_ignores=[]), filename="<ast>", mode="exec"))

# Also helper functions?
funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef) if node.name in ['get_model_based_action']]
for node in funcs:
    exec(compile(ast.Module(body=[node], type_ignores=[]), filename="<ast>", mode="exec"))

print("Classes and Functions loaded.")

def test_environment_randomization():
    print("\n--- Test Environment Randomization ---")
    env = Environment(10, 10)
    starts = set()
    goals = set()
    
    for _ in range(20):
        env.reset()
        starts.add(env.start_pos)
        goals.add(env.goal_pos)
        
        # Check Dist
        dist = abs(env.start_pos[0] - env.goal_pos[0]) + abs(env.start_pos[1] - env.goal_pos[1])
        min_dist = (10 + 10) // 3
        if dist < min_dist:
            print(f"FAIL: Distance {dist} too small (Min: {min_dist})")
            return
            
    if len(starts) > 1 and len(goals) > 1:
        print("PASS: Start and Goal positions vary.")
    else:
        print("FAIL: Start or Goal positions did not vary enough.")

def test_get_observation():
    print("\n--- Test get_observation ---")
    env = Environment(10, 10)
    
    # Fog Mode
    obs_fog = env.get_observation(percept_enabled=True)
    assert obs_fog['mode'] == 'fog'
    assert 'view' in obs_fog
    assert 'grid' not in obs_fog
    print("PASS: Fog Mode returns view only.")
    
    # Full Mode
    obs_full = env.get_observation(percept_enabled=False)
    assert obs_full['mode'] == 'full'
    assert 'grid' in obs_full
    assert 'view' not in obs_full
    print("PASS: Full Mode returns grid.")

def test_q_agent_splitting():
    print("\n--- Test Q-Agent Splitting ---")
    env = Environment(10, 10)
    qa = QAgent(env, 0.5, 0.9, 0.1)
    
    # Train step in Fog
    obs_fog = env.get_observation(percept_enabled=True)
    qa.act(obs_fog)
    qa.post_step(obs_fog, 0)
    qa.learn(obs_fog, 10, obs_fog) # Dummy learn
    
    assert len(qa.q_fog) > 0
    assert np.sum(qa.q_full) == 0
    print("PASS: Fog learning updates q_fog only.")
    
    # Train step in Full
    obs_full = env.get_observation(percept_enabled=False)
    qa.act(obs_full)
    qa.post_step(obs_full, 0)
    qa.learn(obs_full, 10, obs_full)
    
    assert np.sum(qa.q_full) != 0 # Should have updated
    print("PASS: Full learning updates q_full.")

def test_model_based_planner():
    print("\n--- Test Model Based Planner ---")
    grid_mem = {}
    # Simple corridor: S(0,0) -> (0,1) -> G(0,2)
    # Walls at (1,0), (1,1), (1,2)
    start = (0,0)
    goal = (0,2)
    
    # Known free
    grid_mem[(0,0)] = 2 # Start is empty/visited
    # Unknown (0,1) assumed free
    # Known Wall (1,1)
    grid_mem[(1,1)] = 1
    
    action = get_model_based_action(start, goal, grid_mem, 3, 3)
    
    # Should move Right (3) towards (0,1)
    # 0=Up, 1=Down, 2=Left, 3=Right
    if action == 3:
        print("PASS: Planner chose RIGHT towards goal.")
    else:
        print(f"FAIL: Planner chose {action} instead of 3 (Right).")

if __name__ == "__main__":
    test_environment_randomization()
    test_get_observation()
    test_q_agent_splitting()
    test_model_based_planner()
