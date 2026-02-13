from agent_app.environment import Grid
from agent_app.agents import SimpleReflexAgent, ModelBasedReflexAgent, QLearningAgent
import numpy as np

def test_environment():
    print("Testing Environment...")
    env = Grid(5, 5, coverage=0.0)
    assert env.width == 5
    assert env.height == 5
    assert env.grid[0, 0] == 0 # Start
    assert env.grid[4, 4] == 2 # Goal
    print("Environment OK.")

def test_simple_reflex():
    print("Testing Simple Reflex Agent...")
    env = Grid(5, 5, coverage=0.0)
    agent = SimpleReflexAgent()
    percept = env.get_percept()
    action = agent.act(percept)
    assert action in [0, 1, 2, 3]
    print("Simple Reflex Agent OK.")

def test_q_learning():
    print("Testing Q-Learning Agent...")
    env = Grid(5, 5, coverage=0.0)
    agent = QLearningAgent()
    state = env.reset()
    action = agent.act(state)
    next_state, reward, done = env.step(action)
    agent.learn(next_state, reward)
    assert len(agent.q_table) > 0
    print("Q-Learning Agent OK.")

if __name__ == "__main__":
    test_environment()
    test_simple_reflex()
    test_q_learning()
    print("All System Checks Passed.")
