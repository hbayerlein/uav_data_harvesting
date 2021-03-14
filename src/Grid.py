import numpy as np

from src.DeviceManager import DeviceManagerParams, DeviceManager
from src.State import State
from src.base.BaseGrid import BaseGrid, BaseGridParams


class GridParams(BaseGridParams):
    def __init__(self):
        super().__init__()
        self.num_agents_range = [1, 3]
        self.device_manager = DeviceManagerParams()
        self.multi_agent = False
        self.fixed_starting_idcs = False
        self.starting_idcs = [1, 2, 3]


class Grid(BaseGrid):

    def __init__(self, params: GridParams, stats):
        super().__init__(params, stats)
        self.params = params
        if params.multi_agent:
            self.num_agents = params.num_agents_range[0]
        else:
            self.num_agents = 1

        self.device_list = None
        self.device_manager = DeviceManager(self.params.device_manager)

        free_space = np.logical_not(
            np.logical_or(self.map_image.obstacles, self.map_image.start_land_zone))
        free_idcs = np.where(free_space)
        self.device_positions = list(zip(free_idcs[1], free_idcs[0]))

    def get_comm_obstacles(self):
        return self.map_image.obstacles

    def get_data_map(self):
        return self.device_list.get_data_map(self.shape)

    def get_collected_map(self):
        return self.device_list.get_collected_map(self.shape)

    def get_device_list(self):
        return self.device_list

    def get_grid_params(self):
        return self.params

    def init_episode(self):
        self.device_list = self.device_manager.generate_device_list(self.device_positions)

        if self.params.multi_agent:
            self.num_agents = int(np.random.randint(low=self.params.num_agents_range[0],
                                                    high=self.params.num_agents_range[1] + 1, size=1))
        else:
            self.num_agents = 1
        state = State(self.map_image, self.num_agents, self.params.multi_agent)
        state.reset_devices(self.device_list)

        if self.params.fixed_starting_idcs:
            idx = self.params.starting_idcs
        else:
            # Replace False insures that starting positions of the agents are different
            idx = np.random.choice(len(self.starting_vector), size=self.num_agents, replace=False)
        state.positions = [self.starting_vector[i] for i in idx]

        state.movement_budgets = np.random.randint(low=self.params.movement_range[0],
                                                   high=self.params.movement_range[1] + 1, size=self.num_agents)

        state.initial_movement_budgets = state.movement_budgets.copy()

        return state

    def init_scenario(self, scenario):
        self.device_list = scenario.device_list
        self.num_agents = scenario.init_state.num_agents

        return scenario.init_state

    def get_example_state(self):
        if self.params.multi_agent:
            num_agents = self.params.num_agents_range[0]
        else:
            num_agents = 1
        state = State(self.map_image, num_agents, self.params.multi_agent)
        state.device_map = np.zeros(self.shape, dtype=float)
        state.collected = np.zeros(self.shape, dtype=float)
        return state
