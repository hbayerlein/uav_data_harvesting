from src.base.GridActions import GridActions


class GridRewardParams:
    def __init__(self):
        self.boundary_penalty = 1.0
        self.empty_battery_penalty = 150.0
        self.movement_penalty = 0.2


class GridRewards:
    def __init__(self, stats):
        self.params = GridRewardParams()
        self.cumulative_reward: float = 0.0

        stats.add_log_data_callback('cumulative_reward', self.get_cumulative_reward)

    def get_cumulative_reward(self):
        return self.cumulative_reward

    def calculate_motion_rewards(self, state, action: GridActions, next_state):
        reward = 0.0
        if not next_state.landed:
            # Penalize battery Consumption
            reward -= self.params.movement_penalty

        # Penalize not moving (This happens when it either tries to land or fly into a boundary or hovers or fly into
        # a cell occupied by another agent)
        if state.position == next_state.position and not next_state.landed and not action == GridActions.HOVER:
            reward -= self.params.boundary_penalty

        # Penalize battery dead
        if next_state.movement_budget == 0 and not next_state.landed:
            reward -= self.params.empty_battery_penalty

        return reward

    def reset(self):
        self.cumulative_reward = 0
