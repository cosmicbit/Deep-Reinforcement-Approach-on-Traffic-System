import traci
import gym
from gym import spaces
import numpy as np


import save_data


##############################
# SUMO-Based Gym Environment #
##############################
class SUMOTrafficEnv(gym.Env):
    """
    A SUMO-based Gym environment for a four-way intersection.

    This environment assumes you have a SUMO network (compiled from your node and edge files)
    with a central traffic light junction "TL1" and four incoming edges:
      - North: lane ID "edge_n_in_0"
      - South: lane ID "edge_s_in_0"
      - East:  lane ID "edge_e_in_0"
      - West:  lane ID "edge_w_in_0"

    The state vector is [veh_n, veh_s, veh_e, veh_w, current_phase].
    Action space: 0 = keep current phase; 1 = switch to the next phase (cyclic over 4 phases).
    Reward: negative total vehicle count on all incoming lanes.
    """

    def __init__(self, sumo_cmd, max_steps):
        super(SUMOTrafficEnv, self).__init__()
        self.sumo_cmd = sumo_cmd
        self.max_steps = max_steps
        # State: 4 vehicle counts + traffic light phase
        self.observation_space = spaces.Box(low=0, high=100, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)  # Phase action: 0=keep, 1-4 select specific phase.
        self.phase_dict = {
            0: "GGGggrrrrrGGGggrrrrr",
            1: "rrrrrGGGggrrrrrGGGgg"
        }
        # Map the duration_action to a duration value:
        self.step_count = 0
        # self.camera_directions = {
        #     "northbound": (500, 450),  # Watching vehicles coming from the south
        #     "southbound": (500, 550),  # Watching vehicles coming from the north
        #     "eastbound": (550, 500),  # Watching vehicles coming from the west
        #     "westbound": (450, 500)  # Watching vehicles coming from the east
        # }
        # self.detection_range = 50  # 100m detection range

    def reset(self):
        # If TraCI is already connected, close it.
        if traci.isLoaded():
            traci.close()
        traci.start(self.sumo_cmd)
        self.step_count = 0
        # Set initial phase to 0 for traffic light "TL1"
        phase_str = self.phase_dict[1]
        traci.trafficlight.setRedYellowGreenState("TL1", phase_str)
        for i in range(50):
            traci.simulationStep()
            self.step_count += 1
        return self._get_state()

    def _get_state(self):
        trafficlights = traci.trafficlight.getIDList()
        halting_numbers_all = []  # For collecting all halting vehicle counts
        phases_all = []  # For collecting all phases

        for tl_id in trafficlights:
            controlled_lanes = set(traci.trafficlight.getControlledLanes(tl_id))
            edge_ids = set(traci.lane.getEdgeID(lane) for lane in controlled_lanes)  # Sorted for consistency

            edge_ids = list(edge_ids)
            edge_ids.sort()

            halting_numbers = [traci.edge.getLastStepHaltingNumber(edge) for edge in edge_ids]
            halting_numbers_all.extend(halting_numbers)  # Collect halting numbers for normalization

            color_phase = traci.trafficlight.getRedYellowGreenState(tl_id)
            phase = next((k for k, v in self.phase_dict.items() if v == color_phase),
                         -1)  # Default to -1 if phase not found
            phases_all.extend([phase])  # Repeat phase for each halting number
        # Create the state with halting numbers first, followed by phases
        state = halting_numbers_all + phases_all
        # print(state)
        return np.array(state, dtype=np.float32)



    def step(self, action):
        #print("step:", self.step_count)
        self._apply_action(action)
        next_state = self._get_state()
        reward= self._get_reward(next_state)
        done = self.step_count >= self.max_steps
        info = {
            'step_count': self.step_count,
        }
        return next_state, reward, done, info

    def _apply_action(self, action):
        if action != 0:
            color_phase = traci.trafficlight.getRedYellowGreenState("TL1")
            phase = next((k for k, v in self.phase_dict.items() if v == color_phase), -1)
            new_phase = (phase + 1) % 2
            phase_str = self.phase_dict[new_phase]
            traci.trafficlight.setRedYellowGreenState("TL1", phase_str)
        for i in range(30):
            traci.simulationStep()
            self.step_count += 1

    def _get_reward(self, state):
        halting_vehicles = np.array(state[:4])  # First 4 values are halting vehicles
        #moving_vehicles = np.array(state[4:8])  # Next 4 are moving vehicles

        total_halt = np.sum(halting_vehicles)  # Total stopped vehicles
        #total_moving = np.sum(moving_vehicles)  # Total moving vehicles
        std_dev_halt = np.std(halting_vehicles)  # How unbalanced the halt is

        # **Reward Calculation:**
        beta = 0.9  # Weight for standard deviation penalty
        #gamma = 0.5  # Small bonus for moving vehicles

        reward = - total_halt - beta * std_dev_halt #+ gamma * total_moving

        # **Penalty for excessive difference in congestion**
        if np.max(halting_vehicles) - np.min(halting_vehicles) > 30:
            reward -= 100  # Heavy penalty for extreme imbalance

        save_data.save_state_to_csv(state, total_halt, std_dev_halt, beta, reward, filename="data_from_model.csv")
        return reward


    def close(self):
        if traci.isLoaded():
            traci.close()

    """
               def _get_state(self):
        trafficlights = traci.trafficlight.getIDList()
        phases_all = []  # Store traffic light phases
        moving_vehicles_all = []  # Moving vehicles count
        halting_vehicles_all = []  # Halting vehicles count

        for tl_id in trafficlights:
            color_phase = traci.trafficlight.getRedYellowGreenState(tl_id)
            phase = next((k for k, v in self.phase_dict.items() if v == color_phase),
                         -1)  # Default to -1 if phase not found
            phases_all.append(phase)

        # Get data from all 4 cameras
        for cam, pos in self.camera_directions.items():
            moving_count, halting_count = self.get_camera_data(pos)
            moving_vehicles_all.append(moving_count)
            halting_vehicles_all.append(halting_count)

        # Create the state vector: [halting vehicles] + [moving vehicles] + [traffic light phases]
        state = halting_vehicles_all + moving_vehicles_all + phases_all
        return np.array(state, dtype=np.float32)
        
        def get_camera_data(self, camera_pos):
        moving_count = 0  # Vehicles in motion
        halting_count = 0  # Stopped vehicles

        for vehicle_id in traci.vehicle.getIDList():
            x, y = traci.vehicle.getPosition(vehicle_id)  # Get vehicle position
            speed = traci.vehicle.getSpeed(vehicle_id)
            cam_x, cam_y = camera_pos  # Camera coordinates

            # Check if vehicle is within 100m of the camera
            if abs(x - cam_x) <= self.detection_range and abs(y - cam_y) <= self.detection_range:
                if speed == 0:
                    halting_count += 1
                else:
                    moving_count += 1

        return moving_count, halting_count
    """