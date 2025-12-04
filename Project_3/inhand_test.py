# inhand_test.py (Student Skeleton)
import time
# from inhand_env import CanRotateEnv

# # --- TODO: Import your agent class ---
# # from agent import MyRLAgent 

# # --- Configuration ---
# MODEL_PATH = "my_agent_final.pth" # Path to your saved student model
from inhand_env import DiscreteCanRotateEnv
from Intro_to_AI_Students.Project_3.q_agent import QLearningAgent  # <-- adjust module name if needed

# --- Configuration ---
MODEL_PATH = "my_agent_logs/q_table_final.npy"  # match training save path
EPISODES_TO_RUN = 10

# --- TODO: Load the environment ---
# env = CanRotateEnv(render_mode="human")

# --- TODO: Load your trained agent ---
# agent = MyRLAgent(
#     obs_space_shape=env.observation_space.shape,
#     action_space_shape=env.action_space.shape,
#     device='cpu'
# )
# try:
#     agent.load_model(MODEL_PATH)
#     print(f"Successfully loaded model from {MODEL_PATH}")
# except Exception as e:
#     print(f"Error loading model: {e}")
#     exit()


# --- Load the discrete environment ---
env = DiscreteCanRotateEnv(render_mode="human")

# --- Load your trained Q-learning agent ---
agent = QLearningAgent(
    num_states=env.observation_space.n,
    num_actions=env.action_space.n,
)
try:
    agent.load(MODEL_PATH)
    print(f"Successfully loaded Q-table from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading Q-table: {e}")
    env.close()
    raise

# Use a fully greedy policy at test time
agent.epsilon = 0.0



# --- Run the evaluation ---
for episode in range(EPISODES_TO_RUN):
    print(f"--- Starting Episode {episode + 1} ---")
    
    # --- TODO: Reset the environment ---
    # obs, info = env.reset()
    state, info = env.reset()
    
    terminated = False
    truncated = False
    # total_reward = 0
    total_reward = 0.0
    
    while not (terminated or truncated):
        
        # --- TODO: Get a deterministic action from your agent ---
        # The 'deterministic=True' part is key for testing
        # action = agent.get_action(obs, deterministic=True)
        # action = env.action_space.sample() # Placeholder: Replace with your agent's action
        
        # --- TODO: Step the environment ---
        # obs, reward, terminated, truncated, info = env.step(action)
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        state = next_state
        total_reward += reward
        
        # Render/sleep is handled by the environment's step/render methods
        time.sleep(1/60) # Keep visualization smooth
        
    print(f"Episode {episode + 1} finished. Total Reward: {total_reward:.2f}")

# Clean up
env.close()
print("\nEvaluation finished.")