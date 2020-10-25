import os

import numpy as np

from src.IoTDevice import IoTDeviceParams, DeviceList

ColorMap = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]


class DeviceManagerParams:
    def __init__(self):
        self.device_count_range = (2, 5)
        self.data_range = (5.0, 20.0)
        self.fixed_devices = False
        self.devices = IoTDeviceParams()


class DeviceManager:
    """
    The DeviceManager is used to generate DeviceLists according to DeviceManagerParams
    """

    def __init__(self, params: DeviceManagerParams):
        self.params = params

    def generate_device_list(self, positions_vector):
        if self.params.fixed_devices:
            return DeviceList(self.params.devices)

        # Roll number of devices
        device_count = np.random.randint(self.params.device_count_range[0], self.params.device_count_range[1] + 1)

        # Roll Positions
        position_idcs = np.random.choice(range(len(positions_vector)), device_count, replace=False)
        positions = [positions_vector[idx] for idx in position_idcs]

        # Roll Data
        datas = np.random.uniform(self.params.data_range[0], self.params.data_range[1], device_count)

        return self.generate_device_list_from_args(device_count, positions, datas)

    def generate_device_list_from_args(self, device_count, positions, datas):

        # get colors
        colors = ColorMap[0:max(device_count, len(ColorMap))]

        params = [IoTDeviceParams(position=positions[k],
                                  data=datas[k],
                                  color=colors[k % len(ColorMap)])
                  for k in range(device_count)]

        return DeviceList(params)
