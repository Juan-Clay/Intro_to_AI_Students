class DiscreteCanRotateEnv:
    def __init__(self):
        self.env = CanRotateEnv(render_mode="headless")
        self.actions = [...]  # your discrete action vectors
        self.num_states = ...
        self.num_actions = len(self.actions)

    def reset(self):
        obs, info = self.env.reset()
        state = self.obs_to_state(obs)
        return state

    def step(self, action_index):
        action_vec = self.actions[action_index]
        obs, reward, terminated, truncated, info = self.env.step(action_vec)
        next_state = self.obs_to_state(obs)
        done = terminated or truncated
        return next_state, reward, done, info
def obs_to_state(self, obs):
    # extract object pose, convert to yaw angle, bin it
    # optionally check contacts / distance, etc.
    # return an integer in [0, num_states-1]
