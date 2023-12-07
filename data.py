import numpy as np
from dataclasses import dataclass

@dataclass
class DataClass:
    x : int = 1
    y : int = 1
    z : int = 1
    x_counter = 0
    y_counter = 0
    MSD = []
    tumor_vol = np.array([])

    def __post_init__(self):
        self.Abstand = np.zeros((self.x, self.y, self.z))

    def tumor_vol(self, timesteps):
        self.tumor_vol = np.zeros(timesteps)