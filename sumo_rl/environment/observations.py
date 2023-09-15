"""Observation functions for traffic signals."""
from abc import abstractmethod

import numpy as np
from gymnasium import spaces

from .traffic_signal import TrafficSignal
import torch as th


class MyBox(spaces.Box):
    def set_action_mask(self, action_mask):
        self.my_action_mask = action_mask


class ObservationFunction:
    """Abstract base class for observation functions."""

    def __init__(self, ts: TrafficSignal):
        """Initialize observation function."""
        self.ts = ts

    @abstractmethod
    def __call__(self):
        """Subclasses must override this method."""
        pass

    @abstractmethod
    def observation_space(self):
        """Subclasses must override this method."""
        pass


class DefaultObservationFunction(ObservationFunction):
    """Default observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize default observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.ts.get_lanes_density()
        queue = self.ts.get_lanes_queue()
        observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        obs_space = MyBox(
            low=np.zeros(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
        )
        action_mask = th.ones(8)
        if len(self.ts.sorted_incoming_roads) == 3:
            action_mask[0:5] = 0
        elif len(self.ts.sorted_incoming_roads) == 2:
            action_mask[0:4] = 0
            action_mask[6:8] = 0
        obs_space.set_action_mask(action_mask)
        return obs_space


class FullAttachObservationFunction(ObservationFunction):
    """Full lane (attached with lanes) observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize default observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.ts.get_lanes_density()
        out_density = self.ts.get_out_lanes_density()
        queue = self.ts.get_lanes_queue()
        out_queue = self.ts.get_out_lanes_queue()
        observation = np.array(phase_id + min_green + density + out_density + queue + out_queue, dtype=np.float32)
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 1 + 2 * (len(self.ts.lanes) + len(self.ts.out_lanes)),
                         dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 1 + 2 * (len(self.ts.lanes) + len(self.ts.out_lanes)),
                         dtype=np.float32),
        )


class FullClockwiseObservationFunction(ObservationFunction):
    """Clockwised lane observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize default observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.ts.get_clock_wise_lanes_density()
        queue = self.ts.get_clock_wise_lanes_queue()
        observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 1 + 2 * (len(self.ts.clock_wise_lanes) + len(self.ts.out_lanes)),
                         dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 1 + 2 * (len(self.ts.clock_wise_lanes) + len(self.ts.out_lanes)),
                         dtype=np.float32),
        )
