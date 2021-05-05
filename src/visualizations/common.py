
import numpy as np

def spherical_coordinates(x, y, z):

    latitude = np.arcsin(z)
    longitude = np.arctan2(y, x)

    return latitude, longitude


def hammer_atioff(latitude, longitude):

    z = np.sqrt(1 + np.cos(latitude) * np.cos(longitude / 2))
    x = 2 * np.cos(latitude) * np.sin(longitude / 2) / z
    y = np.sin(latitude) / z

    return x, y
