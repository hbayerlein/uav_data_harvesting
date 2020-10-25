import numpy as np
from src.Map.Shadowing import load_or_create_shadowing


class ChannelParams:
    def __init__(self):
        self.cell_edge_snr = -25  # in dB
        self.los_path_loss_exp = 2.27
        self.nlos_path_loss_exp = 3.64
        self.uav_altitude = 10.0  # in m
        self.cell_size = 10.0  # in m
        self.los_shadowing_variance = 2.0
        self.nlos_shadowing_variance = 5.0
        self.map_path = "res/manhattan32.png"


class Channel:
    def __init__(self, params: ChannelParams):
        self.params = params
        self._norm_distance = None
        self.los_norm_factor = None
        self.los_shadowing_sigma = None
        self.nlos_shadowing_sigma = None
        self.total_shadow_map = load_or_create_shadowing(self.params.map_path)

    def reset(self, area_size):
        self._norm_distance = np.sqrt(2) * 0.5 * area_size * self.params.cell_size
        self.los_norm_factor = 10 ** (self.params.cell_edge_snr / 10) / (
                self._norm_distance ** (-self.params.los_path_loss_exp))
        self.los_shadowing_sigma = np.sqrt(self.params.los_shadowing_variance)
        self.nlos_shadowing_sigma = np.sqrt(self.params.nlos_shadowing_variance)

    def get_max_rate(self):
        dist = self.params.uav_altitude

        snr = self.los_norm_factor * dist ** (-self.params.los_path_loss_exp)

        rate = np.log2(1 + snr)

        return rate

    def compute_rate(self, uav_pos, device_pos):
        dist = np.sqrt(
            ((device_pos[0] - uav_pos[0]) * self.params.cell_size) ** 2 +
            ((device_pos[1] - uav_pos[1]) * self.params.cell_size) ** 2 +
            self.params.uav_altitude ** 2)

        if self.total_shadow_map[int(round(device_pos[1])), int(round(device_pos[0])),
                                 int(round(uav_pos[1])), int(round(uav_pos[0]))]:
            snr = self.los_norm_factor * dist ** (
                -self.params.nlos_path_loss_exp) * 10 ** (np.random.normal(0., self.nlos_shadowing_sigma) / 10)
        else:
            snr = self.los_norm_factor * dist ** (
                -self.params.los_path_loss_exp) * 10 ** (np.random.normal(0., self.los_shadowing_sigma) / 10)

        rate = np.log2(1 + snr)

        return rate
