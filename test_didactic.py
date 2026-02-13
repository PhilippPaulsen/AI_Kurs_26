import sys
import types
import numpy as np

# Mock streamlit with expanded capabilities for new features
st = types.ModuleType("streamlit")
class SessionState(dict):
    def __getattr__(self, item): return self.get(item)
    def __setattr__(self, key, value): self[key] = value

st.session_state = SessionState()
st.set_page_config = lambda **kwargs: None
st.markdown = lambda *args, **kwargs: None
st.write = lambda *args, **kwargs: None
st.expander = lambda *args, **kwargs: MockExpander()
st.dataframe = lambda *args, **kwargs: None

# Mock Sidebar
st.sidebar = types.ModuleType("streamlit.sidebar")
st.sidebar.title = lambda *args: None
st.sidebar.slider = lambda *args, **kwargs: kwargs.get('value', 10) 
st.sidebar.selectbox = lambda *args, **kwargs: "Manuell"
st.sidebar.checkbox = lambda *args, **kwargs: True
st.sidebar.button = lambda *args, **kwargs: False
st.sidebar.markdown = lambda *args: None
st.sidebar.subheader = lambda *args: None
class MockExpander:
    def __enter__(self): return self
    def __exit__(self, *args): pass

st.sidebar.expander = lambda *args, **kwargs: MockExpander()

# Mock Components & Columns
class MockColumn:
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def button(self, *args, **kwargs): return False
    def metric(self, *args, **kwargs): pass

def mock_columns(spec):
    n = spec if isinstance(spec, int) else len(spec)
    return [MockColumn() for _ in range(n)]
st.columns = mock_columns
st.empty = lambda: types.SimpleNamespace(markdown=lambda *args, **kwargs: None)
st.button = lambda *args, **kwargs: False
st.components = types.ModuleType("streamlit.components")
st.components.v1 = types.ModuleType("streamlit.components.v1")
st.components.v1.html = lambda *args, **kwargs: None

sys.modules["streamlit"] = st
sys.modules["streamlit.components.v1"] = st.components.v1

# Import main
sys.path.append('/Users/philipp.paulsen/Documents/iu/kurse/2026_L/AI/AI_Kurs_26/agent_app')
from main import Environment, QAgent

def test_environment_config():
    # Test custom penalty/reward
    env = Environment(10, 10, step_penalty=-0.5, goal_reward=50.0)
    assert env.step_penalty == -0.5
    assert env.goal_reward == 50.0
    
    # Test step returns penalty
    env.agent_pos = (1, 1) # Force valid start
    _, r, _ = env.step(1) # Move Right
    assert r == -0.5, f"Expected step penalty -0.5, got {r}"
    
    print("✅ Environment Config test passed")

def test_q_agent_logging():
    env = Environment(5, 5)
    agent = QAgent(env, alpha=0.5, gamma=0.9, epsilon=0.1)
    
    # Simulate a step and learn
    s = (1, 1)
    a = 1 # DOWN (in grid logic 0=Up, 1=Down?) check main.py logic: moves = [(-1,0), (1,0), (0,-1), (0,1)] (Up, Down, Left, Right)
    # 0: Up, 1: Down, 2: Left, 3: Right
    
    agent.post_step(s, a)
    s_next = (2, 1)
    r = -0.1
    
    st.session_state.logs = [] # Clear logs
    agent.learn(s, r, s_next)
    
    assert len(st.session_state.logs) > 0, "Log should not be empty after learn()"
    last_log = st.session_state.logs[-1]
    assert "Q-Update" in last_log, "Log should contain Q-Update info"
    assert "Q_neu" in last_log, "Log should contain math details"
    
    print("✅ Q-Agent Logging test passed")

if __name__ == "__main__":
    try:
        test_environment_config()
        test_q_agent_logging()
        print("\nALL DIDACTIC TESTS PASSED")
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
