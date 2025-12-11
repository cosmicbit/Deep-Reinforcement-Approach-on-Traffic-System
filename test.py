# test_sumo.py
import numpy as np
import torch
from sumoenv import SUMOTrafficEnv
from agent import  DQNAgent  # Import classes from your training module

import ttl
from visualization import Visualization


def test_agent(sumo_cmd, episodes):
    """
    Runs the trained agent for a given number of episodes in the SUMO simulation,
    using SUMO-gui so you can visually inspect the simulation.
    """
    env = SUMOTrafficEnv(sumo_cmd, max_steps=1000)
    state_dim = env.observation_space.shape[0]  # Now 9
    # Compute composite action dimension from the environment's action space:
    action_n = env.action_space.n  # e.g., 5
    # Create a new DQN agent instance
    loaded_agent = DQNAgent(state_dim, action_n)

    # Load the saved model parameters (ensure the file "dqn_sumo_model.pth" exists)
    loaded_agent.q_network.load_state_dict(torch.load("models/dqn_fourway_model.pth", map_location=loaded_agent.device))
    loaded_agent.update_target_network()
    loaded_agent.epsilon = 0.0  # Ensure no exploration during testing

    loaded_agent.epsilon = 0.0  # Set to greedy policy (no exploration)
    for ep in range(episodes):
        state = env.reset()
        rewards=[]
        done = False
        prev_step_count=-100
        while not done:
            state_tensor = torch.FloatTensor(state).to(loaded_agent.device)
            with torch.no_grad():
                action = int(torch.argmax(loaded_agent.q_network(state_tensor)).item())
            # action = loaded_agent.unflatten_action(flat_action)
            state, reward, done, info = env.step(action)
            step = info['step_count']
            rewards.append(reward)
            if prev_step_count+100 <= step:
                print(f"Step {step}: State = {state}, Reward = {reward}")
                prev_step_count = step
        print(f"Test Episode {ep + 1} completed.")
        print(f"Total reward of episode {ep + 1}: {np.sum(rewards)}")
        print(f"Mean reward of episode {ep + 1}: {np.mean(rewards)}")
    env.close()


    viz = Visualization()
    # Plot the reward at each step
    viz.save_data_and_plot(data=rewards, filename='ROT testing', xlabel='timesteps', ylabel='Cumulative negative reward')


if __name__ == "__main__":
    # Define the SUMO command in GUI mode (adjust the config file name if needed)
    sumo_cmd = ["sumo-gui", "-c", "one_intersection/simple_intersection.sumocfg"]


    # Run one or more test episodes
    test_agent(sumo_cmd, episodes=1)
    ttl.run_simulation(sumo_cmd, total_steps=1000, print_interval=50)
