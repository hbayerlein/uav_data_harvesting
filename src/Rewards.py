from src.State import State
from src.base.GridActions import GridActions
from src.base.GridRewards import GridRewards, GridRewardParams


class RewardParams(GridRewardParams):
    def __init__(self):
        super().__init__()
        self.data_multiplier = 1.0


# Class used to track rewards
class Rewards(GridRewards):
    cumulative_reward: float = 0.0

    def __init__(self, reward_params: RewardParams, stats):
        super().__init__(stats)
        self.params = reward_params
        self.reset()

    def calculate_reward(self, state: State, action: GridActions, next_state: State):
        reward = self.calculate_motion_rewards(state, action, next_state)

        # Reward the collected data
        reward += self.params.data_multiplier * (state.get_remaining_data() - next_state.get_remaining_data())

        # Cumulative reward
        self.cumulative_reward += reward

        return reward
