import numpy as np
from skimage import io


class Map:
    def __init__(self, map_data):
        self.start_land_zone = map_data[:, :, 2].astype(bool)
        self.nfz = map_data[:, :, 0].astype(bool)
        self.obstacles = map_data[:, :, 1].astype(bool)

    def get_starting_vector(self):
        similar = np.where(self.start_land_zone)
        return list(zip(similar[1], similar[0]))

    def get_free_space_vector(self):
        free_space = np.logical_not(
            np.logical_or(self.obstacles, self.start_land_zone))
        free_idcs = np.where(free_space)
        return list(zip(free_idcs[1], free_idcs[0]))

    def get_size(self):
        return self.start_land_zone.shape[:2]


def load_image(path):
    if type(path) is not str:
        raise TypeError('path needs to be a string')
    data = io.imread(path, as_gray=True)
    return np.array(data, dtype=bool)


def save_image(path, image):
    if type(path) is not str:
        raise TypeError('path needs to be a string')
    if image.dtype == bool:
        io.imsave(path, image * np.uint8(255))
    else:
        io.imsave(path, image)


def load_map(path):
    if type(path) is not str:
        raise TypeError('path needs to be a string')
    data = io.imread(path, as_gray=False)
    return Map(data)


def load_target(path):
    if type(path) is not str:
        raise TypeError('path needs to be a string')
    data = io.imread(path, as_gray=True)
    return np.array(data, dtype=bool)
