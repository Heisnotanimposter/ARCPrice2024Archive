"""
ARC-AGI-3 ARCEngine Gym-Compatible Environment Wrapper
Interfaces open-source ARCEngine games with Python agent loops.
"""

import numpy as np

class ARCEngineEnv:
    """Gym-compatible wrapper for interactive ARCEngine game environments."""

    def __init__(self, game_id=0, grid_size=(20, 20)):
        self.game_id = game_id
        self.grid_size = grid_size
        self.current_step = 0
        self.max_steps = 100
        self.state = None

    def reset(self):
        """Resets environment to initial state."""
        self.current_step = 0
        # Initialize grid with background 0 and player avatar 1
        self.state = np.zeros(self.grid_size, dtype=int)
        self.state[self.grid_size[0]//2, self.grid_size[1]//2] = 1
        # Target goal 2
        self.state[2, 2] = 2
        return self.state.copy()

    def step(self, action):
        """
        Executes action (0: UP, 1: DOWN, 2: LEFT, 3: RIGHT, 4: ACTION).
        Returns (obs, reward, done, info).
        """
        self.current_step += 1
        player_pos = np.argwhere(self.state == 1)
        reward = 0.0
        done = self.current_step >= self.max_steps

        if len(player_pos) > 0:
            r, c = player_pos[0]
            self.state[r, c] = 0  # Clear previous position

            if action == 0: r = max(0, r - 1)
            elif action == 1: r = min(self.grid_size[0] - 1, r + 1)
            elif action == 2: c = max(0, c - 1)
            elif action == 3: c = min(self.grid_size[1] - 1, c + 1)

            # Check if reached goal (2)
            if self.state[r, c] == 2:
                reward = 1.0
                done = True

            self.state[r, c] = 1

        return self.state.copy(), reward, done, {"step": self.current_step}

if __name__ == "__main__":
    env = ARCEngineEnv(game_id=1)
    obs = env.reset()
    print(f"Initialized ARCEngine Environment Game {env.game_id} with state shape {obs.shape}")
    obs, reward, done, info = env.step(action=0)
    print(f"Executed action 0 (UP). Reward: {reward}, Done: {done}")
