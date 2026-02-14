
import sys
import os

# Add the directory to path to import agent_app modules
sys.path.append(os.getcwd())

try:
    from agent_app.main import QAgent, Environment
    print("Successfully imported QAgent and Environment")
    
    env = Environment()
    agent = QAgent(env, 0.5, 0.9, 0.1)
    
    if hasattr(agent, 'last_td_error'):
        print(f"agent.last_td_error exists and is: {agent.last_td_error}")
    else:
        print("ERROR: agent.last_td_error does NOT exist")
        exit(1)
        
    # simulate a learning step
    agent.prev_s = (1,1)
    agent.prev_a = 0
    agent.learn((1,1), 10, (1,2))
    
    print(f"agent.last_td_error after learn: {agent.last_td_error}")
    
    if agent.last_td_error != 0.0:
        print("TD-Error updated successfully")
    else:
        print("Warning: TD-Error is still 0.0 (might be correct if values were 0, but checking logic)")

except Exception as e:
    print(f"An error occurred: {e}")
    exit(1)
