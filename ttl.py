import os
import sys
import gym
import numpy as np
from gym import spaces
import save_data
from visualization import Visualization

# Set SUMO_HOME and update system path (adjust the path if needed)
if "SUMO_HOME" not in os.environ:
    os.environ["SUMO_HOME"] = "/usr/share/sumo"
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
import traci


class FixedTimeEnv(gym.Env):
    """
    A SUMO-based Gym environment that uses SUMO's built-in fixed-time (traditional) traffic light logic.

    Assumes a four-way intersection with:
      - Four incoming lanes with IDs: "edge_n_in_0", "edge_s_in_0", "edge_e_in_0", and "edge_w_in_0".
      - A traffic light junction "TL1" defined in your network file.

    The state vector is:
        [veh_n, veh_s, veh_e, veh_w, current_phase]
    where veh_* is the vehicle count on the corresponding incoming lane,
    and current_phase is the current phase of the traffic light.

    Since the fixed-time controller does not intervene (the tlLogic is predefined),
    no actions are taken. The simulation simply advances.

    Reward (optional): negative sum of vehicle counts (for logging purposes).
    """

    def __init__(self, sumo_cmd, max_steps):
        super(FixedTimeEnv, self).__init__()
        self.sumo_cmd = sumo_cmd
        self.max_steps = max_steps
        self.observation_space = spaces.Box(low=0, high=100, shape=(5,), dtype=np.float32)
        # Action space is dummy; no actions will be taken in fixed-time control.
        self.action_space = spaces.Discrete(2)
        self.phase_dict = {
            0: "ggggrrrrggggrrrr",
            1: "rrrrggggrrrrgggg"
        }
        self.step_count = 0
        # self.camera_directions = {
        #     "northbound": (500, 450),  # Watching vehicles coming from the south
        #     "southbound": (500, 550),  # Watching vehicles coming from the north
        #     "eastbound": (550, 500),  # Watching vehicles coming from the west
        #     "westbound": (450, 500)  # Watching vehicles coming from the east
        # }
        # self.detection_range = 50  # 100m detection range

    def reset(self):
        if traci.isLoaded():
            traci.close()
        traci.start(self.sumo_cmd)
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

    def step(self, action=None):
        traci.simulationStep()
        self.step_count += 1
        next_state = self._get_state()
        reward = self._get_reward(next_state)
        done = self.step_count >= self.max_steps

        return next_state, reward, done, {}

    def _get_reward(self, state):
        halting_vehicles = np.array(state[:4])  # First 4 values are halting vehicles
        # moving_vehicles = np.array(state[4:8])  # Next 4 are moving vehicles

        total_halt = np.sum(halting_vehicles)  # Total stopped vehicles
        # total_moving = np.sum(moving_vehicles)  # Total moving vehicles
        std_dev_halt = np.std(halting_vehicles)  # How unbalanced the halt is

        # **Reward Calculation:**
        beta = 0.9  # Weight for standard deviation penalty
        # gamma = 0.5  # Small bonus for moving vehicles

        reward = - total_halt - beta * std_dev_halt  # + gamma * total_moving

        # **Penalty for excessive difference in congestion**
        if np.max(halting_vehicles) - np.min(halting_vehicles) > 30:
            reward -= 100  # Heavy penalty for extreme imbalance
        save_data.save_state_to_csv(state, total_halt, std_dev_halt, beta, reward,
                                    filename="data_from_ttl.csv")
        return reward

    def close(self):
        if traci.isLoaded():
            traci.close()


def run_simulation(sumo_cmd, total_steps, print_interval):
    env = FixedTimeEnv(sumo_cmd, max_steps=total_steps)
    state = env.reset()
    rewards=[]
    print("Starting Fixed-Time (TTL) Simulation")
    for step in range(total_steps):
        state, reward, done, _ = env.step()
        rewards.append(reward)
        if step % print_interval == 0:
            print(f"Step {step}: State = {state}, Reward = {reward}")
        if done:
            break
        # Sleep a little if you want to slow down the simulation
        #time.sleep(0.05)
    env.close()
    print("Simulation Completed.")
    print(f"Total reward of simulation : {np.sum(rewards)}")
    print(f"Mean reward of simulation: {np.mean(rewards)}")
    viz = Visualization()
    # Plot the reward at each step
    viz.save_data_and_plot(data=rewards, filename='ROT TTL', xlabel='timesteps', ylabel='Cumulative negative reward')



if __name__ == "__main__":
    # Set the SUMO command to use SUMO-gui (for visualization)
    # Ensure that "fourway_intersection.sumocfg" exists in your working directory.
    sumo_cmd = ["sumo-gui", "-c", "one_intersection/simple_intersection.sumocfg"]
    run_simulation(sumo_cmd, total_steps=1000, print_interval=50)
