"""
ARC-AGI-3 Rule Intuition Agent
Autonomous agent for uninstructed ARCEngine game exploration and rule inference.
"""

import random
import numpy as np

class RuleIntuitionAgent:
    """Agent that explores ARCEngine environments, builds transition rules, and selects actions."""

    def __init__(self, action_space_size=5):
        self.action_space_size = action_space_size
        self.transition_history = []

    def select_action(self, observation):
        """Chooses action based on current state observation and exploration policy."""
        player_pos = np.argwhere(observation == 1)
        goal_pos = np.argwhere(observation == 2)

        if len(player_pos) > 0 and len(goal_pos) > 0:
            pr, pc = player_pos[0]
            gr, gc = goal_pos[0]

            # Heuristic directional navigation towards inferred goal
            if pr > gr: return 0  # UP
            if pr < gr: return 1  # DOWN
            if pc > gc: return 2  # LEFT
            if pc < gc: return 3  # RIGHT

        return random.randint(0, self.action_space_size - 1)

    def update_rules(self, prev_obs, action, next_obs, reward):
        """Records state transition to update world model hypotheses."""
        self.transition_history.append({
            "prev_obs": prev_obs,
            "action": action,
            "next_obs": next_obs,
            "reward": reward
        })

if __name__ == "__main__":
    from ARC_AGI_3.environments.arc_engine_env import ARCEngineEnv
except ImportError:
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from environments.arc_engine_env import ARCEngineEnv

if __name__ == "__main__":
    env = ARCEngineEnv()
    agent = RuleIntuitionAgent()

    obs = env.reset()
    total_reward = 0
    for step in range(50):
        action = agent.select_action(obs)
        next_obs, reward, done, info = env.step(action)
        agent.update_rules(obs, action, next_obs, reward)
        obs = next_obs
        total_reward += reward
        if done:
            print(f"Game finished in {step+1} steps! Total Reward: {total_reward}")
            break
