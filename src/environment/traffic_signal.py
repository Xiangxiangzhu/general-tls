"""This module contains the TrafficSignal class, which represents a traffic signal in the simulation."""
import os
import sys
import sumolib
import math
from typing import Callable, List, Union
import numpy as np
from gymnasium import spaces
from operator import itemgetter
import pandas as pd

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")


class TrafficSignal:
    """This class represents a Traffic Signal controlling an intersection.

    It is responsible for retrieving information and changing the traffic phase using the Traci API.

    IMPORTANT: It assumes that the traffic phases defined in the .net file are of the form:
        [green_phase, yellow_phase, green_phase, yellow_phase, ...]
    Currently it is not supporting all-red phases (but should be easy to implement it).

    # Observation Space
    The default observation for each traffic signal agent is a vector:

    obs = [phase_one_hot, min_green, lane_1_density,...,lane_n_density, lane_1_queue,...,lane_n_queue]

    - ```phase_one_hot``` is a one-hot encoded vector indicating the current active green phase
    - ```min_green``` is a binary variable indicating whether min_green seconds have already passed in the current phase
    - ```lane_i_density``` is the number of vehicles in incoming lane i dividided by the total capacity of the lane
    - ```lane_i_queue``` is the number of queued (speed below 0.1 m/s) vehicles in incoming lane i divided by the total capacity of the lane

    You can change the observation space by implementing a custom observation class. See :py:class:`src.environment.observations.ObservationFunction`.

    # Action Space
    Action space is discrete, corresponding to which green phase is going to be open for the next delta_time seconds.

    # Reward Function
    The default reward function is 'diff-waiting-time'. You can change the reward function by implementing a custom reward function and passing to the constructor of :py:class:`src.environment.env.SumoEnvironment`.
    """

    # Default min gap of SUMO (see https://sumo.dlr.de/docs/Simulation/Safety.html). Should this be parameterized?
    MIN_GAP = 2.5

    def __init__(
            self,
            env,
            ts_id: str,
            delta_time: int,
            yellow_time: int,
            min_green: int,
            max_green: int,
            begin_time: int,
            reward_fn: Union[str, Callable],
            sumo,
    ):
        """Initializes a TrafficSignal object.

        Args:
            env (SumoEnvironment): The environment this traffic signal belongs to.
            ts_id (str): The id of the traffic signal.
            delta_time (int): The time in seconds between actions.
            yellow_time (int): The time in seconds of the yellow phase.
            min_green (int): The minimum time in seconds of the green phase.
            max_green (int): The maximum time in seconds of the green phase.
            begin_time (int): The time in seconds when the traffic signal starts operating.
            reward_fn (Union[str, Callable]): The reward function. Can be a string with the name of the reward function or a callable function.
            sumo (Sumo): The Sumo instance.
        """
        self.id = ts_id
        self.env = env
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.green_phase = 0
        self.is_yellow = False
        self.time_since_last_phase_change = 0
        self.next_action_time = begin_time
        self.last_measure = 0.0
        self.last_reward = None
        self.reward_fn = reward_fn
        self.sumo = sumo

        if type(self.reward_fn) is str:
            if self.reward_fn in TrafficSignal.reward_fns.keys():
                self.reward_fn = TrafficSignal.reward_fns[self.reward_fn]
            else:
                raise NotImplementedError(f"Reward function {self.reward_fn} not implemented")

        self.observation_fn = self.env.observation_class(self)

        self.lanes = list(
            dict.fromkeys(self.sumo.trafficlight.getControlledLanes(self.id))
        )  # Remove duplicates and keep order
        self.out_lanes = [link[0][1] for link in self.sumo.trafficlight.getControlledLinks(self.id) if link]
        self.out_lanes = list(set(self.out_lanes))
        self.lanes_length = {lane: self.sumo.lane.getLength(lane) for lane in self.lanes + self.out_lanes}

        self.all_lanes = self.lanes + self.out_lanes

        nd = NetworkData(self.env._net)
        self.netdata = nd.get_net_data()

        # reorder incoming roads
        self.all_roads = [self.netdata['lane'][lane]["edge"] for lane in self.all_lanes]
        self.all_roads = list(set(self.all_roads))
        self.incoming_roads = [self.netdata['lane'][lane]["edge"] for lane in self.lanes]
        self.incoming_roads = list(set(self.incoming_roads))

        # sort tls controlled lanes in clock wise
        self.clock_wise_lanes = self._reshape_index()

        # build all green phases and yellow phases
        self._build_phases()

        self.observation_space = self.observation_fn.observation_space()
        self.action_space = spaces.Discrete(self.num_green_phases)

    @staticmethod
    def custom_sort_rule(char):
        # order = {'s': 2, 'r': 1, 'l': 3}
        order = {'r': 1, 'R': 2, 's': 3, 'L': 4, 'l': 5, 't': 6}
        return order.get(char, 0)

    def _reshape_index(self):
        print("##### reshaped lane tsc index #####")

        heading = {}
        for road in self.all_roads:
            if road in self.incoming_roads:
                road_from = self.netdata['edge'][road]['coord'][0]
                road_to = self.netdata['edge'][road]['coord'][1]
            else:
                road_from = self.netdata['edge'][road]['coord'][1]
                road_to = self.netdata['edge'][road]['coord'][0]
            angle_x = road_from[0] - road_to[0]
            angle_y = road_from[1] - road_to[1]
            theta = math.atan2(angle_y, angle_x)
            degrees = math.degrees(theta)
            adjusted_degrees = (90 - degrees) % 360
            # print("road is ", road)
            # print("theta is ", adjusted_degrees)
            heading[road] = adjusted_degrees

        sorted_heading = sorted(heading.items(), key=itemgetter(1))
        self.sorted_dict = {k: v for k, v in sorted_heading}

        sorted_road_data = sorted(
            sorted_heading,
            key=lambda x: (x[1], -self.incoming_roads.index(x[0]) if x[0] in self.incoming_roads else float('inf'))
        )
        self.clock_wise_all_roads = [item[0] for item in sorted_road_data]
        self.clock_wise_incoming_roads = [item for item in self.clock_wise_all_roads if item in self.incoming_roads]

        # create tsc index Dataframe : merged_df
        tls_index = self.netdata["inter"][self.id]["tlsindex"]
        tls_index_dir = self.netdata["inter"][self.id]["tlsindexdir"]

        def remove_negative(tls_dict):
            temp_to_remove = []
            for idx in tls_dict:
                if idx < 0:
                    temp_to_remove.append(idx)
            for idx in temp_to_remove:
                del tls_dict[idx]
            return tls_dict

        tls_index = remove_negative(tls_index)
        tls_index_dir = remove_negative(tls_index_dir)

        df1 = pd.DataFrame(list(tls_index.items()), columns=['tsl_index', 'lane_name'])
        df2 = pd.DataFrame(list(tls_index_dir.items()), columns=['tsl_index', 'lane_dir'])

        # concatenate DataFrame
        merged_df = pd.concat([df1, df2['lane_dir']], axis=1)
        self.merged_df = merged_df

        def is_consecutive_increment(sequence):
            n = len(sequence)
            if n <= 1:
                return True
            for i in range(1, n):
                if sequence[i] != sequence[i - 1] + 1:
                    return False
            return True

        # get the origin tsl index of a intersection, it should be a monotonically increasing sequence with a step of 1
        self.tsl_index_origin = sorted(merged_df["tsl_index"].tolist())
        # print("self.tsl_index_origin is ", self.tsl_index_origin)
        assert is_consecutive_increment(self.tsl_index_origin), "tsl index error !!!!"

        merged_df['reordered_index'] = [-1 for _ in range(len(merged_df))]

        # reorder tsc index
        temp_id = min(self.tsl_index_origin)
        for road in self.sorted_dict:
            lane_reverse = False
            lane_list = self.netdata["edge"][road]["lanes"]
            if "r" in self.netdata["lane"][lane_list[0]]['movement'] or "l" in self.netdata["lane"][lane_list[-1]][
                'movement']:
                lane_reverse = False
            elif "l" in self.netdata["lane"][lane_list[0]]['movement'] or "r" in self.netdata["lane"][lane_list[-1]][
                'movement']:
                lane_reverse = True
            if lane_reverse:
                lane_list.reverse()
            for lane_ in lane_list:
                if lane_ in merged_df['lane_name'].tolist():
                    temp_dir = self.netdata["lane"][lane_]['movement']
                    sorted_lane_dir = ''.join(sorted(temp_dir, key=self.custom_sort_rule))
                    for d in sorted_lane_dir:
                        matching_keys = int(merged_df.loc[(merged_df['lane_name'] == lane_) & (
                                merged_df['lane_dir'] == d), 'tsl_index'].values[0])
                        merged_df.loc[merged_df['tsl_index'] == matching_keys, 'reordered_index'] = temp_id
                        temp_id += 1

        df_sorted = merged_df.sort_values('reordered_index')
        reordered_lanes = df_sorted['lane_name'].tolist()
        # self.reordered_tsl_index = df_sorted
        return reordered_lanes

    def _build_green_phases(self):
        # generate phase for this inter
        road_phases = {}

        self.sorted_incoming_roads = {r: self.sorted_dict[r] for r in self.sorted_dict if r in self.incoming_roads}

        for road, lane_data in self.sorted_incoming_roads.items():
            movement_temp = "".join([self.netdata['lane'][l]["movement"] for l in self.netdata["edge"][road]['lanes']])
            counts = {char: movement_temp.count(char) for char in ["r", "R", "s", "l", "L", "t"]}

            l_t_phase = (counts["r"] + counts["R"]) * "g" + counts["s"] * "r" + (counts["l"] + counts["L"]) * "G" + \
                        counts["t"] * "g"
            s_t_phase = (counts["r"] + counts["R"]) * "g" + counts["s"] * "G" + (counts["l"] + counts["L"]) * "r" + \
                        counts["t"] * "r"
            l_s_phase = (counts["r"] + counts["R"]) * "g" + counts["s"] * "G" + (counts["l"] + counts["L"]) * "G" + \
                        counts["t"] * "g"
            stop_phase = (counts["r"] + counts["R"] + counts["s"] + counts["l"] + counts["L"] + counts["t"]) * "r"

            road_phases[road] = [l_t_phase, s_t_phase, l_s_phase, stop_phase]

        phase_patterns = {
            4: [
                [0, 1, 3, 3, 2, 3, 3, 3],
                [3, 3, 0, 1, 3, 2, 3, 3],
                [0, 1, 3, 3, 3, 3, 2, 3],
                [3, 3, 0, 1, 3, 3, 3, 2]
            ],
            3: [
                [3, 3, 3, 3, 3, 2, 3, 3],
                [3, 3, 3, 3, 3, 3, 2, 3],
                [3, 3, 3, 3, 3, 3, 3, 2]
            ],
            2: [
                [3, 3, 3, 3, 2, 3, 3, 3],
                [3, 3, 3, 3, 3, 2, 3, 3]
            ]
        }

        num_roads = len(self.sorted_incoming_roads)
        phases = [""] * 8

        for idx, road in enumerate(self.sorted_incoming_roads):
            for i, phase_idx in enumerate(phase_patterns[num_roads][idx]):
                phases[i] += road_phases[road][phase_idx]

        self.phase_inter = phases
        self.regulated_green_phases = [self.sumo.trafficlight.Phase(60, phase) for phase in self.phase_inter]
        return self.regulated_green_phases

    def _build_phases(self):
        # temp_phases = self.sumo.trafficlight.getAllProgramLogics(self.id)
        # phases = self.sumo.trafficlight.getAllProgramLogics(self.id)[0].phases

        self.yellow_dict = {}
        self.green_phases = self._build_green_phases()
        self.num_green_phases = len(self.green_phases)
        self.all_phases = self.green_phases.copy()

        for i, p1 in enumerate(self.green_phases):
            for j, p2 in enumerate(self.green_phases):
                if i == j:
                    continue
                yellow_state = ""
                for s in range(len(p1.state)):
                    if (p1.state[s] == "G" or p1.state[s] == "g") and (p2.state[s] == "r" or p2.state[s] == "s"):
                        yellow_state += "y"
                    else:
                        yellow_state += p1.state[s]
                self.yellow_dict[(i, j)] = len(self.all_phases)
                self.all_phases.append(self.sumo.trafficlight.Phase(self.yellow_time, yellow_state))

        programs = self.sumo.trafficlight.getAllProgramLogics(self.id)
        logic = programs[0]
        logic.type = 0
        logic.phases = self.all_phases
        self.sumo.trafficlight.setProgramLogic(self.id, logic)
        self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[0].state)

    @property
    def time_to_act(self):
        """Returns True if the traffic signal should act in the current step."""
        return self.next_action_time == self.env.sim_step

    def update(self):
        """Updates the traffic signal state.

        If the traffic signal should act, it will set the next green phase and update the next action time.
        """
        self.time_since_last_phase_change += 1
        if self.is_yellow and self.time_since_last_phase_change == self.yellow_time:
            # self.sumo.trafficlight.setPhase(self.id, self.green_phase)
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
            self.is_yellow = False

    def set_next_phase(self, new_phase: int):
        """Sets what will be the next green phase and sets yellow phase if the next phase is different than the current.

        Args:
            new_phase (int): Number between [0 ... num_green_phases]
        """
        new_phase = int(new_phase)
        if self.green_phase == new_phase or self.time_since_last_phase_change < self.yellow_time + self.min_green:
            # self.sumo.trafficlight.setPhase(self.id, self.green_phase)
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
            self.next_action_time = self.env.sim_step + self.delta_time
        else:
            # self.sumo.trafficlight.setPhase(self.id, self.yellow_dict[(self.green_phase, new_phase)])  # turns yellow
            self.sumo.trafficlight.setRedYellowGreenState(
                self.id, self.all_phases[self.yellow_dict[(self.green_phase, new_phase)]].state
            )
            self.green_phase = new_phase
            self.next_action_time = self.env.sim_step + self.delta_time
            self.is_yellow = True
            self.time_since_last_phase_change = 0

    def compute_observation(self):
        """Computes the observation of the traffic signal."""
        return self.observation_fn()

    def compute_reward(self):
        """Computes the reward of the traffic signal."""
        self.last_reward = self.reward_fn(self)
        return self.last_reward

    def _pressure_reward(self):
        return self.get_pressure()

    def _average_speed_reward(self):
        return self.get_average_speed()

    def _queue_reward(self):
        return -self.get_total_queued()

    def _diff_waiting_time_reward(self):
        ts_wait = sum(self.get_accumulated_waiting_time_per_lane()) / 100.0
        reward = self.last_measure - ts_wait
        self.last_measure = ts_wait
        return reward

    def _observation_fn_default(self):
        phase_id = [1 if self.green_phase == i else 0 for i in range(self.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.time_since_last_phase_change < self.min_green + self.yellow_time else 1]
        density = self.get_lanes_density()
        queue = self.get_lanes_queue()
        observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
        return observation

    def get_accumulated_waiting_time_per_lane(self) -> List[float]:
        """Returns the accumulated waiting time per lane.

        Returns:
            List[float]: List of accumulated waiting time of each intersection lane.
        """
        wait_time_per_lane = []
        for lane in self.lanes:
            veh_list = self.sumo.lane.getLastStepVehicleIDs(lane)
            wait_time = 0.0
            for veh in veh_list:
                veh_lane = self.sumo.vehicle.getLaneID(veh)
                acc = self.sumo.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles[veh] = {veh_lane: acc}
                else:
                    self.env.vehicles[veh][veh_lane] = acc - sum(
                        [self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane]
                    )
                wait_time += self.env.vehicles[veh][veh_lane]
            wait_time_per_lane.append(wait_time)
        return wait_time_per_lane

    def get_average_speed(self) -> float:
        """Returns the average speed normalized by the maximum allowed speed of the vehicles in the intersection.

        Obs: If there are no vehicles in the intersection, it returns 1.0.
        """
        avg_speed = 0.0
        vehs = self._get_veh_list()
        if len(vehs) == 0:
            return 1.0
        for v in vehs:
            avg_speed += self.sumo.vehicle.getSpeed(v) / self.sumo.vehicle.getAllowedSpeed(v)
        return avg_speed / len(vehs)

    def get_pressure(self):
        """Returns the pressure (#veh leaving - #veh approaching) of the intersection."""
        return sum(self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.out_lanes) - sum(
            self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.lanes
        )

    def get_out_lanes_density(self) -> List[float]:
        """Returns the density of the vehicles in the outgoing lanes of the intersection."""
        lanes_density = [
            self.sumo.lane.getLastStepVehicleNumber(lane)
            / (self.lanes_length[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane)))
            for lane in self.out_lanes
        ]
        return [min(1, density) for density in lanes_density]

    def get_lanes_density(self) -> List[float]:
        """Returns the density [0,1] of the vehicles in the incoming lanes of the intersection.

        Obs: The density is computed as the number of vehicles divided by the number of vehicles that could fit in the lane.
        """
        lanes_density = [
            self.sumo.lane.getLastStepVehicleNumber(lane)
            / (self.lanes_length[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane)))
            for lane in self.lanes
        ]
        return [min(1, density) for density in lanes_density]

    def get_lanes_density_fn(self, lanes) -> List[float]:
        """Returns the density [0,1] of the vehicles in the incoming lanes of the intersection.

        Obs: The density is computed as the number of vehicles divided by the number of vehicles that could fit in the lane.
        """
        lanes_density = [
            self.sumo.lane.getLastStepVehicleNumber(lane)
            / (self.lanes_length[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane)))
            for lane in lanes
        ]
        return [min(1, density) for density in lanes_density]

    def get_lanes_queue(self) -> List[float]:
        """Returns the queue [0,1] of the vehicles in the incoming lanes of the intersection.

        Obs: The queue is computed as the number of vehicles halting divided by the number of vehicles that could fit in the lane.
        """
        lanes_queue = [
            self.sumo.lane.getLastStepHaltingNumber(lane)
            / (self.lanes_length[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane)))
            for lane in self.lanes
        ]
        return [min(1, queue) for queue in lanes_queue]

    def get_lanes_queue_fn(self, lanes) -> List[float]:
        lanes_queue = [
            self.sumo.lane.getLastStepHaltingNumber(lane)
            / (self.lanes_length[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane)))
            for lane in lanes
        ]
        return [min(1, queue) for queue in lanes_queue]

    def get_out_lanes_queue(self) -> List[float]:
        """Returns the queue [0,1] of the vehicles in the incoming lanes of the intersection.

        Obs: The queue is computed as the number of vehicles halting divided by the number of vehicles that could fit in the lane.
        """
        lanes_queue = [
            self.sumo.lane.getLastStepHaltingNumber(lane)
            / (self.lanes_length[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane)))
            for lane in self.out_lanes
        ]
        return [min(1, queue) for queue in lanes_queue]

    def get_total_queued(self) -> int:
        """Returns the total number of vehicles halting in the intersection."""
        return sum(self.sumo.lane.getLastStepHaltingNumber(lane) for lane in self.lanes)

    def get_clock_wise_lanes_density(self) -> List[float]:
        """Returns the density [0,1] of the vehicles in all lanes of the intersection sorted clock-wise.

        Obs: The density is computed as the number of vehicles divided by the number of vehicles that could fit in the lane.
        """
        lanes_density = [
            self.sumo.lane.getLastStepVehicleNumber(lane)
            / (self.lanes_length[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane)))
            for lane in self.clock_wise_lanes
        ]
        return [min(1, density) for density in lanes_density]

    def get_clock_wise_lanes_queue(self) -> List[float]:
        """Returns the queue [0,1] of the vehicles in the all lanes of the intersection sorted clock-wise.

        Obs: The queue is computed as the number of vehicles halting divided by the number of vehicles that could fit in the lane.
        """
        lanes_queue = [
            self.sumo.lane.getLastStepHaltingNumber(lane)
            / (self.lanes_length[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane)))
            for lane in self.clock_wise_lanes
        ]
        return [min(1, queue) for queue in lanes_queue]

    def _get_veh_list(self):
        veh_list = []
        for lane in self.lanes:
            veh_list += self.sumo.lane.getLastStepVehicleIDs(lane)
        return veh_list

    @classmethod
    def register_reward_fn(cls, fn: Callable):
        """Registers a reward function.

        Args:
            fn (Callable): The reward function to register.
        """
        if fn.__name__ in cls.reward_fns.keys():
            raise KeyError(f"Reward function {fn.__name__} already exists")

        cls.reward_fns[fn.__name__] = fn

    reward_fns = {
        "diff-waiting-time": _diff_waiting_time_reward,
        "average-speed": _average_speed_reward,
        "queue": _queue_reward,
        "pressure": _pressure_reward,
    }


class NetworkData:
    def __init__(self, net_fp):
        print(net_fp)
        self.net = sumolib.net.readNet(net_fp)
        ###get edge data
        self.edge_data = self.get_edge_data(self.net)
        self.lane_data = self.get_lane_data(self.net)
        self.node_data, self.intersection_data = self.get_node_data(self.net)
        print("SUCCESSFULLY GENERATED NET DATA")

    def get_net_data(self):
        return {'lane': self.lane_data, 'edge': self.edge_data, 'origin': self.find_origin_edges(),
                'destination': self.find_destination_edges(), 'node': self.node_data, 'inter': self.intersection_data}

    # inter
    def find_destination_edges(self):
        # next_edges = {e: 0 for e in self.edge_data}
        next_edges = {}
        for e in self.edge_data:
            if self.edge_data[e]['allow']:
                next_edges[e] = 0

        for e in self.edge_data:
            if self.edge_data[e]['allow']:
                for next_e in self.edge_data[e]['incoming']:
                    if self.edge_data[next_e]['allow']:
                        next_edges[next_e] += 1

        destinations = [e for e in next_edges if next_edges[e] == 0]
        return destinations

    # inter
    def find_origin_edges(self):
        # next_edges = {e: 0 for e in self.edge_data}
        next_edges = {}
        for e in self.edge_data:
            if self.edge_data[e]['allow']:
                next_edges[e] = 0

        for e in self.edge_data:
            for next_e in self.edge_data[e]['outgoing']:
                next_edges[next_e] += 1

        origins = [e for e in next_edges if next_edges[e] == 0]
        return origins

    # inter
    def get_edge_data(self, net):
        edges = net.getEdges()
        edge_data = {str(edge.getID()): {} for edge in edges}

        for edge in edges:
            edge_ID = str(edge.getID())
            edge_data[edge_ID]['lanes'] = [str(lane.getID()) for lane in edge.getLanes()]
            edge_data[edge_ID]['length'] = float(edge.getLength())
            # edge_data[edge_ID]['outgoing'] = [str(out.getID()) for out in edge.getOutgoing()]
            edge_data[edge_ID]['outgoing'] = [str(out.getID()) for out in edge.getAllowedOutgoing("private")]
            edge_data[edge_ID]['noutgoing'] = len(edge_data[edge_ID]['outgoing'])
            edge_data[edge_ID]['nlanes'] = len(edge_data[edge_ID]['lanes'])
            edge_data[edge_ID]['incoming'] = [str(inc.getID()) for inc in edge.getIncoming()]
            edge_data[edge_ID]['outnode'] = str(edge.getFromNode().getID())
            edge_data[edge_ID]['incnode'] = str(edge.getToNode().getID())
            edge_data[edge_ID]['speed'] = float(edge.getSpeed())
            edge_data[edge_ID]['allow'] = edge.allows("private")

            ###coords for each edge
            incnode_coord = edge.getFromNode().getCoord()
            outnode_coord = edge.getToNode().getCoord()
            edge_data[edge_ID]['coord'] = np.array(
                [incnode_coord[0], incnode_coord[1], outnode_coord[0], outnode_coord[1]]).reshape(2, 2)
            # print edge_data[edge_ID]['coord']
        return edge_data

    # inter
    def get_lane_data(self, net):
        # create lane objects from lane_ids
        lane_ids = []
        for edge in self.edge_data:
            lane_ids.extend(self.edge_data[edge]['lanes'])

        lanes = [net.getLane(lane) for lane in lane_ids]
        # lane data dict
        lane_data = {lane: {} for lane in lane_ids}

        for lane in lanes:
            lane_id = lane.getID()
            lane_data[lane_id]['length'] = lane.getLength()
            lane_data[lane_id]['speed'] = lane.getSpeed()
            lane_data[lane_id]['edge'] = str(lane.getEdge().getID())
            # lane_data[ lane_id ]['outgoing'] = []
            lane_data[lane_id]['outgoing'] = {}
            ###.getOutgoing() returns a Connection type, which we use to determine the next lane
            moveid = []
            for conn in lane.getOutgoing():
                out_id = str(conn.getToLane().getID())
                lane_data[lane_id]['outgoing'][out_id] = {'dir': str(conn.getDirection()),
                                                          'index': conn.getTLLinkIndex()}
                moveid.append(str(conn.getDirection()))
            lane_data[lane_id]['movement'] = ''.join(sorted(moveid))
            # create empty list for incoming lanes
            lane_data[lane_id]['incoming'] = []

        # determine incoming lanes using outgoing lanes data
        for lane in lane_data:
            for inc in lane_data:
                if lane == inc:
                    continue
                else:
                    if inc in lane_data[lane]['outgoing']:
                        lane_data[inc]['incoming'].append(lane)

        return lane_data

    # inter
    def get_node_data(self, net):
        # read network from .netfile
        nodes = net.getNodes()
        node_data = {node.getID(): {} for node in nodes}

        for node in nodes:
            node_id = node.getID()
            # get incoming/outgoing edge
            node_data[node_id]['incoming'] = set(str(edge.getID()) for edge in node.getIncoming())
            node_data[node_id]['outgoing'] = set(str(edge.getID()) for edge in node.getOutgoing())
            node_data[node_id]['tlsindex'] = {conn.getTLLinkIndex(): str(conn.getFromLane().getID()) for conn in
                                              node.getConnections()}
            node_data[node_id]['tlsindexdir'] = {conn.getTLLinkIndex(): str(conn.getDirection()) for conn in
                                                 node.getConnections()}

            if node_id == '-13968':
                print("##### what is -13968 ????")
                missing = []
                negative = []
                for i in range(len(node_data[node_id]['tlsindex'])):
                    if i not in node_data[node_id]['tlsindex']:
                        missing.append(i)

                for k in node_data[node_id]['tlsindex']:
                    if k < 0:
                        negative.append(k)

                for m, n in zip(missing, negative):
                    node_data[node_id]['tlsindex'][m] = node_data[node_id]['tlsindex'][n]
                    del node_data[node_id]['tlsindex'][n]
                    # for index dir
                    node_data[node_id]['tlsindexdir'][m] = node_data[node_id]['tlsindexdir'][n]
                    del node_data[node_id]['tlsindexdir'][n]

            # get XY coords
            pos = node.getCoord()
            node_data[node_id]['x'] = pos[0]
            node_data[node_id]['y'] = pos[1]

        intersection_data = {str(node): node_data[node] for node in node_data if
                             "traffic_light" in net.getNode(node).getType()}

        return node_data, intersection_data
