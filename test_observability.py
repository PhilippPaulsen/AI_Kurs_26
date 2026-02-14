
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
funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef) if node.name in ['get_model_based_action', 'render_grid_html']]
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

# test_observability removed (ReflexAgent logic is inline in main.py)

def test_get_observation():
    print("\n--- Test get_observation ---")
    env = Environment(10, 10)
    
    # 1. Full Mode
    obs_full = env.get_observation("full")
    if 'grid' in obs_full and obs_full['mode'] == 'full' and obs_full['goal_pos'] is not None:
        print("PASS: Full Mode returns grid and goal.")
    else:
        print("FAIL: Full Mode missing keys:", obs_full.keys())
        failed = True
        
    # 2. Fog Mode
    obs_fog = env.get_observation("fog")
    if 'view' in obs_fog and 'neighbors' in obs_fog and obs_fog.get('agent_pos') is not None:
         print("PASS: Fog Mode returns view, neighbors and agent_pos.")
    else:
        print("FAIL: Fog Mode behavior incorrect:", obs_fog.keys())
        failed = True

    # 3. Strict Mode
    obs_strict = env.get_observation("strict")
    if 'view' in obs_strict and 'neighbors' in obs_strict and obs_strict.get('agent_pos') is None:
        print("PASS: Strict Mode returns view/neighbors and HIDES agent_pos.")
    else:
        print("FAIL: Strict Mode behavior incorrect:", obs_strict.keys())
        failed = True

def test_q_agent_splitting():
    print("\n--- Test Q-Agent Splitting ---")
    env = Environment(10, 10)
    qa = QAgent(env, 0.5, 0.9, 0.1)
    
    # Train step in FOG
    obs = env.get_observation("fog")
    qa.prev_state_key = qa.encode_state(obs)
    qa.prev_action = 0
    next_obs = env.get_observation("fog")
    qa.learn(obs, 10, next_obs)
    
    if len(qa.q_fog) > 0 and np.sum(qa.q_full) == 0:
        print("PASS: Fog learning updates q_fog only.")
    else:
        print("FAIL: Fog learning leaked to q_full or didn't update.")
        failed = True
        
    # Train step in FULL
    obs = env.get_observation("full")
    qa.prev_state_key = qa.encode_state(obs)
    qa.prev_action = 0
    next_obs = env.get_observation("full")
    qa.learn(obs, 10, next_obs)
    
    if np.sum(qa.q_full) != 0:
        print("PASS: Full learning updates q_full.")
    else:
        print("FAIL: Full learning didn't update q_full.")
        failed = True

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

def test_q_agent_act_obs():
    print("\n--- Test Q-Agent Act with Obs ---")
    env = Environment(10, 10)
    qa = QAgent(env, 0.5, 0.9, 0.1)
    
    # Test Full Mode
    obs_full = env.get_observation("full")
    try:
        action = qa.act(obs_full)
        assert action in [0, 1, 2, 3]
        print("PASS: Q-Agent acts correctly with Full Obs.")
    except Exception as e:
        print(f"FAIL: Q-Agent failed with Full Obs: {e}")

    # Test Fog Mode
    obs_fog = env.get_observation("fog")
    try:
        action = qa.act(obs_fog)
        assert action in [0, 1, 2, 3]
        print("PASS: Q-Agent acts correctly with Fog Obs.")
    except Exception as e:
        print(f"FAIL: Q-Agent failed with Fog Obs: {e}")

def test_heatmap_attributes():
    print("\n--- Test Heatmap Attributes ---")
    env = Environment(10, 10)
    qa = QAgent(env, 0.5, 0.9, 0.1)
    
    if hasattr(qa, 'q_full'):
         print("PASS: Q-Agent has q_full attribute for heatmap.")
    else:
         print("FAIL: Q-Agent missing q_full.")

def test_fog_heatmap_projection():
    print("\n--- Test Fog Heatmap Projection Logic ---")
    env = Environment(10, 10)
    qa = QAgent(env, 0.5, 0.9, 0.1)
    
    # Manually train a bit in Fog
    obs = env.get_observation("fog")
    qa.act(obs)
    qa.post_step(obs, 0)
    qa.learn(obs, 10, obs)
    
    # Try to project
    try:
        q_grid = np.zeros((env.height, env.width))
        # Just test one cell
        r, c = 5, 5
        # Synthesize View
        syn_view = {}
        radius = 1
        for i in range(-radius, radius+1):
            for j in range(-radius, radius+1):
                if abs(i) + abs(j) <= radius:
                    nr, nc = r+i, c+j
                    val = -1
                    if 0 <= nr < env.height and 0 <= nc < env.width:
                        val = env.grid[nr, nc]
                    syn_view[(nr, nc)] = val
        
        # Calculate Neighbors for synthetic obs
        neighbors = []
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
             nr, nc = r+dr, c+dc
             val = syn_view.get((nr, nc), -1)
             neighbors.append(val)
        
        syn_obs = {
            'mode': 'fog',
            'agent_pos': (r, c),
            'goal_pos': env.goal_pos,
            'view': syn_view,
            'neighbors': tuple(neighbors),
            'is_game_over': False
        }
        
        state_key = qa.encode_state(syn_obs)
        vals = qa.get_q(state_key)
        assert len(vals) == 4
        print("PASS: Fog Heatmap projection logic works.")
    except Exception as e:
        print(f"FAIL: Fog Heatmap projection failed: {e}")

def test_render_grid_html():
    print("\n--- Test render_grid_html ---")
    env = Environment(10, 10)
    qa = QAgent(env, 0.5, 0.9, 0.1)
    
    # Test Fog Mode Render
    try:
        # Note: function signature updated or logic internally updated?
        # The function signature was `render_grid_html(env, agent_type, percept_enabled, strict_fog=False, q_agent=None)`
        # It relies on `env.get_observation`.
        # Wait, `render_grid_html` uses `env.get_percept_view`.
        # I didn't verify `render_grid_html` updates.
        # But let's assume it works or fails.
        html = render_grid_html(env, "Q-Learning", percept_enabled=True, q_agent=qa)
        assert isinstance(html, str)
        print("PASS: render_grid_html works in Fog Mode.")
    except Exception as e:
        print(f"FAIL: render_grid_html failed in Fog Mode: {e}")

    # Test Full Mode Render
    try:
        html = render_grid_html(env, "Q-Learning", percept_enabled=False, strict_fog=False, q_agent=qa)
        assert isinstance(html, str)
        print("PASS: render_grid_html works in Full Mode.")
    except Exception as e:
         # Might fail if args mismatch
        print(f"FAIL: render_grid_html failed in Full Mode: {e}")

def test_strict_fog_behavior():
    print("\n--- Test Strict Fog Behavior ---")
    env = Environment(10, 10)
    
    # 1. Test Get Observation
    obs_strict = env.get_observation("strict")
    assert obs_strict['goal_pos'] is None or obs_strict['goal_pos'] == env.goal_pos # Might be visible if close
    assert obs_strict['agent_pos'] is None
    assert 'neighbors' in obs_strict
    
    # 2. Test Q-Agent Encoding
    qa = QAgent(env, 0.5, 0.9, 0.1)
    
    state_strict = qa.encode_state(obs_strict)
    # ('strict', neighbors) -> Length 2
    assert len(state_strict) == 2
    assert state_strict[0] == 'strict'
    print("PASS: Q-Agent encodes Strict Fog correctly (Local Only).")

if __name__ == "__main__":
    test_environment_randomization()
    # test_observability() # Removed
    test_get_observation()
    test_q_agent_splitting()
    test_model_based_planner()
    test_q_agent_act_obs()
    test_heatmap_attributes()
    test_fog_heatmap_projection()
    test_render_grid_html()
    test_strict_fog_behavior()

if __name__ == "__main__":
    test_environment_randomization()
    test_get_observation()
    test_q_agent_splitting()
    test_model_based_planner()
    test_q_agent_act_obs()
    test_heatmap_attributes()
    test_fog_heatmap_projection()
    test_render_grid_html()
    test_strict_fog_behavior()
