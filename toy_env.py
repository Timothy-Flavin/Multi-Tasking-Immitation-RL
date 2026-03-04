import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ContextualDecouplerEnv(gym.Env):
    """
    A minimal Gymnasium environment to test factored credit assignment.
    The environment randomly switches which action head is 'important' and
    penalizes the 'unimportant' head for taking non-zero actions.
    """
    metadata = {"render_modes": []}

    def __init__(self, n_actions=5, max_steps=100):
        super().__init__()
        self.n_actions = n_actions
        self.max_steps = max_steps
        self.current_step = 0
        
        # Action space: Two independent discrete heads
        self.action_space = spaces.MultiDiscrete([n_actions, n_actions])
        
        # Observation space: [Context (0 or 1), Target 0, Target 1]
        self.observation_space = spaces.MultiDiscrete([2, n_actions, n_actions])
        
        self._state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self._state = self.observation_space.sample()
        return self._state, {}

    def step(self, action):
        context, target_0, target_1 = self._state
        act_0, act_1 = action
        
        reward = 0.0
        
        # Evaluate actions based on the active context
        if context == 0:
            # Head 0 is in control
            reward += 1.0 if act_0 == target_0 else -1.0
            
            # Penalize Head 1 for acting out of turn (0 is the 'do nothing' baseline)
            if act_1 != 0:
                reward -= 0.1
        else:
            # Head 1 is in control
            reward += 1.0 if act_1 == target_1 else -1.0
            
            # Penalize Head 0 for acting out of turn
            if act_0 != 0:
                reward -= 0.1
                
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        # Generate new state for the next step (completely randomized context and targets)
        self._state = self.observation_space.sample()
        
        return self._state, reward, terminated, truncated, {}

# Optional: Register the environment if you are using gym.make()
# gym.envs.registration.register(
#     id='ContextualDecoupler-v0',
#     entry_point='__main__:ContextualDecouplerEnv',
# )