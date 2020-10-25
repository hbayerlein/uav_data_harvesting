import numpy as np

from src.Channel import Channel


class IoTDeviceParams:
    def __init__(self, position=(0, 0), color='blue', data=15.0):
        self.position = position
        self.data = data
        self.color = color


class IoTDevice:
    data: float
    collected_data: float
    # data_timeseries = []
    # data_rate_timeseries = []

    def __init__(self, params: IoTDeviceParams):
        self.params = params

        self.position = params.position  # fixed position can be later overwritten in reset
        self.color = params.color

        self.data = params.data
        # self.data_timeseries = [self.data]
        # self.data_rate_timeseries = [0]
        self.collected_data = 0

    def collect_data(self, collect):
        if collect == 0:
            return 1
        c = min(collect, self.data - self.collected_data)
        self.collected_data += c

        # return collection ratio, i.e. the percentage of time used for comm
        return c / collect

    @property
    def depleted(self):
        return self.data <= self.collected_data

    def get_data_rate(self, pos, channel: Channel):
        rate = channel.compute_rate(uav_pos=pos, device_pos=self.position)
        # self.data_rate_timeseries.append(rate)
        return rate

    # def log_data(self):
    #     self.data_timeseries.append(self.data - self.collected_data)


class DeviceList:

    def __init__(self, params):
        self.devices = [IoTDevice(device) for device in params]

    def get_data_map(self, shape):
        data_map = np.zeros(shape, dtype=float)

        for device in self.devices:
            data_map[device.position[1], device.position[0]] = device.data - device.collected_data

        return data_map

    def get_collected_map(self, shape):
        data_map = np.zeros(shape, dtype=float)

        for device in self.devices:
            data_map[device.position[1], device.position[0]] = device.collected_data

        return data_map

    def get_best_data_rate(self, pos, channel: Channel):
        """
        Get the best data rate and the corresponding device index
        """
        data_rates = np.array(
            [device.get_data_rate(pos, channel) if not device.depleted else 0 for device in self.devices])
        idx = np.argmax(data_rates) if data_rates.any() else -1
        return data_rates[idx], idx

    def collect_data(self, collect, idx):
        ratio = 1
        if idx != -1:
            ratio = self.devices[idx].collect_data(collect)

        # for device in self.devices:
        #     device.log_data()

        return ratio

    def get_devices(self):
        return self.devices

    def get_device(self, idx):
        return self.devices[idx]

    def get_total_data(self):
        return sum(list([device.data for device in self.devices]))

    def get_collected_data(self):
        return sum(list([device.collected_data for device in self.devices]))

    @property
    def num_devices(self):
        return len(self.devices)
