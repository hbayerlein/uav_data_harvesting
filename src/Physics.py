import numpy as np

from src.Channel import ChannelParams, Channel
from src.State import State
from src.ModelStats import ModelStats
from src.base.GridActions import GridActions
from src.base.GridPhysics import GridPhysics


class PhysicsParams:
    def __init__(self):
        self.channel_params = ChannelParams()
        self.comm_steps = 4


class Physics(GridPhysics):

    def __init__(self, params: PhysicsParams, stats: ModelStats):

        super().__init__()

        self.channel = Channel(params.channel_params)

        self.params = params

        self.register_functions(stats)

    def register_functions(self, stats: ModelStats):
        stats.set_evaluation_value_callback(self.get_cral)

        stats.add_log_data_callback('cral', self.get_cral)
        stats.add_log_data_callback('cr', self.get_collection_ratio)
        stats.add_log_data_callback('successful_landing', self.has_landed)
        stats.add_log_data_callback('boundary_counter', self.get_boundary_counter)
        stats.add_log_data_callback('landing_attempts', self.get_landing_attempts)
        stats.add_log_data_callback('movement_ratio', self.get_movement_ratio)

    def reset(self, state: State):
        GridPhysics.reset(self, state)

        self.channel.reset(self.state.shape[0])

    def step(self, action: GridActions):
        old_position = self.state.position
        self.movement_step(action)
        if not self.state.terminal:
            self.comm_step(old_position)

        return self.state

    def comm_step(self, old_position):
        positions = list(
            reversed(np.linspace(self.state.position, old_position, num=self.params.comm_steps, endpoint=False)))

        indices = []
        device_list = self.state.device_list
        for position in positions:
            data_rate, idx = device_list.get_best_data_rate(position, self.channel)
            device_list.collect_data(data_rate, idx)
            indices.append(idx)

        self.state.collected = device_list.get_collected_map(self.state.shape)
        self.state.device_map = device_list.get_data_map(self.state.shape)

        idx = max(set(indices), key=indices.count)
        self.state.set_device_com(idx)

        return idx

    def get_example_action(self):
        return GridActions.HOVER

    def is_in_landing_zone(self):
        return self.state.is_in_landing_zone()

    def get_collection_ratio(self):
        return self.state.get_collection_ratio()

    def get_movement_budget_used(self):
        return sum(self.state.initial_movement_budgets) - sum(self.state.movement_budgets)

    def get_max_rate(self):
        return self.channel.get_max_rate()

    def get_average_data_rate(self):
        return self.state.get_collected_data() / self.get_movement_budget_used()

    def get_cral(self):
        return self.get_collection_ratio() * self.state.all_landed

    def get_boundary_counter(self):
        return self.boundary_counter

    def get_landing_attempts(self):
        return self.landing_attempts

    def get_movement_ratio(self):
        return float(self.get_movement_budget_used()) / float(sum(self.state.initial_movement_budgets))

    def has_landed(self):
        return self.state.all_landed
