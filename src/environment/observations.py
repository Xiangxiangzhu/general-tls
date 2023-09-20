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
        density = self.get_general_state("density")
        queue = self.get_general_state("queue")
        observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
        return observation

    def get_general_state(self, func):
        # get lanes in each road
        # calculate r, s, l list for each road
        # form general state
        assert func in ["density", "queue"], "plz define calculation function!!!"
        if func is "density":
            metric = self.ts.get_lanes_density_fn
        elif func is "queue":
            metric = self.ts.get_lanes_queue_fn

        lane_dict = {road: self.ts.netdata['edge'][road]['lanes'] for road in self.ts.clock_wise_incoming_roads}

        road_wise_state = {}
        for road_id, lane in lane_dict.items():
            r_turn_veh, s_turn_veh, l_turn_veh = [0], [0], [0]
            for l_ in lane:
                if "r" in self.ts.netdata['lane'][l_]["movement"] or "R" in self.ts.netdata['lane'][l_]["movement"]:
                    r_turn_veh.append(metric([l_])[0])
                if "s" in self.ts.netdata['lane'][l_]["movement"]:
                    s_turn_veh.append(metric([l_])[0])
                if ("l" in self.ts.netdata['lane'][l_]["movement"] or "L" in self.ts.netdata['lane'][l_]["movement"]
                        or "t" in self.ts.netdata['lane'][l_]["movement"]):
                    l_turn_veh.append(metric([l_])[0])

            lane_state = [max(r_turn_veh), max(s_turn_veh), max(l_turn_veh)]
            road_wise_state[road_id] = lane_state

        return [item for sublist in road_wise_state.values() for item in sublist]

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        obs_space = MyBox(
            low=np.zeros(self.ts.num_green_phases + 1 + 2 * 3 * len(self.ts.incoming_roads), dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 1 + 2 * 3 * len(self.ts.incoming_roads), dtype=np.float32),
        )
        action_mask = th.ones(8)
        if len(self.ts.sorted_incoming_roads) == 3:
            action_mask[0:4] = 0
            action_mask[5:6] = 0
        elif len(self.ts.sorted_incoming_roads) == 2:
            action_mask[0:4] = 0
            action_mask[6:8] = 0
        obs_space.set_action_mask(action_mask)
        return obs_space


class FullAttachObservationFunction(ObservationFunction):
    """Default observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize default observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.get_general_state("density")
        queue = self.get_general_state("queue")
        observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
        return observation

    def get_general_state(self, func):
        assert func in ["density", "queue"], "plz define calculation function!!!"
        if func is "density":
            metric = self.ts.get_lanes_density_fn
        elif func is "queue":
            metric = self.ts.get_lanes_queue_fn

        attached_roads = self.ts.clock_wise_incoming_roads + [item for item in self.ts.clock_wise_all_roads if
                                                              item not in self.ts.clock_wise_incoming_roads]
        lane_dict = {road: self.ts.netdata['edge'][road]['lanes'] for road in attached_roads}

        road_wise_state = {}
        for road_id, lane in lane_dict.items():
            r_turn_veh, s_turn_veh, l_turn_veh = [0], [0], [0]
            for l_ in lane:
                if "r" in self.ts.netdata['lane'][l_]["movement"] or "R" in self.ts.netdata['lane'][l_]["movement"]:
                    r_turn_veh.append(metric([l_])[0])
                if "s" in self.ts.netdata['lane'][l_]["movement"]:
                    s_turn_veh.append(metric([l_])[0])
                if ("l" in self.ts.netdata['lane'][l_]["movement"] or "L" in self.ts.netdata['lane'][l_]["movement"]
                        or "t" in self.ts.netdata['lane'][l_]["movement"]):
                    l_turn_veh.append(metric([l_])[0])

            lane_state = [max(r_turn_veh), max(s_turn_veh), max(l_turn_veh)]
            road_wise_state[road_id] = lane_state

        return [item for sublist in road_wise_state.values() for item in sublist]

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        obs_space = MyBox(
            low=np.zeros(self.ts.num_green_phases + 1 + 2 * len(self.ts.clock_wise_all_roads), dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 1 + 2 * len(self.ts.clock_wise_all_roads), dtype=np.float32),
        )
        action_mask = th.ones(8)
        if len(self.ts.sorted_incoming_roads) == 3:
            action_mask[0:5] = 0
        elif len(self.ts.sorted_incoming_roads) == 2:
            action_mask[0:4] = 0
            action_mask[6:8] = 0
        obs_space.set_action_mask(action_mask)
        return obs_space


class FullClockwiseObservationFunction(ObservationFunction):
    """Default observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize default observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.get_general_state("density")
        queue = self.get_general_state("queue")
        observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
        return observation

    def get_general_state(self, func):
        assert func in ["density", "queue"], "plz define calculation function!!!"
        if func is "density":
            metric = self.ts.get_lanes_density_fn
        elif func is "queue":
            metric = self.ts.get_lanes_queue_fn

        lane_dict = {road: self.ts.netdata['edge'][road]['lanes'] for road in self.ts.clock_wise_all_roads}

        road_wise_state = {}
        for road_id, lane in lane_dict.items():
            r_turn_veh, s_turn_veh, l_turn_veh = [0], [0], [0]
            for l_ in lane:
                if "r" in self.ts.netdata['lane'][l_]["movement"] or "R" in self.ts.netdata['lane'][l_]["movement"]:
                    r_turn_veh.append(metric([l_])[0])
                if "s" in self.ts.netdata['lane'][l_]["movement"]:
                    s_turn_veh.append(metric([l_])[0])
                if ("l" in self.ts.netdata['lane'][l_]["movement"] or "L" in self.ts.netdata['lane'][l_]["movement"]
                        or "t" in self.ts.netdata['lane'][l_]["movement"]):
                    l_turn_veh.append(metric([l_])[0])

            lane_state = [max(r_turn_veh), max(s_turn_veh), max(l_turn_veh)]
            road_wise_state[road_id] = lane_state

        return [item for sublist in road_wise_state.values() for item in sublist]

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        obs_space = MyBox(
            low=np.zeros(self.ts.num_green_phases + 1 + 2 * len(self.ts.clock_wise_all_roads), dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 1 + 2 * len(self.ts.clock_wise_all_roads), dtype=np.float32),
        )
        action_mask = th.ones(8)
        if len(self.ts.sorted_incoming_roads) == 3:
            action_mask[0:5] = 0
        elif len(self.ts.sorted_incoming_roads) == 2:
            action_mask[0:4] = 0
            action_mask[6:8] = 0
        obs_space.set_action_mask(action_mask)
        return obs_space
