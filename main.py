import os
import sys
import datetime
import torch

from agent import DQNAgent
from sumoenv import SUMOTrafficEnv
from visualization import Visualization

# Set SUMO_HOME and add SUMO tools to path.
if "SUMO_HOME" not in os.environ:
    os.environ["SUMO_HOME"] = "/usr/share/sumo"  # Adjust path if necessary
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))




#############
# Training #
#############

def train_dqn(sumo_cmd, episodes, batch_size, target_update_freq=10):
    env = SUMOTrafficEnv(sumo_cmd, max_steps=1000)
    state_dim = env.observation_space.shape[0]  # Now 9
    # Compute composite action dimension from the environment's action space:
    action_n = env.action_space.n  # e.g., 2
    # duration_n = env.action_space.spaces[1].n  # e.g., 3
    action_dim = action_n  # 3*3 = 9
    agent = DQNAgent(state_dim, action_dim)
    print("Initial epsilon:", agent.epsilon)
    all_rewards = []
    timestamp_start = datetime.datetime.now()
    for ep in range(episodes):
        state = env.reset()
        #print("Episode", ep)
        ep_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            #print("Selected Action", action_tuple)
            # Flatten the action tuple for storage/training.
            # flat_action = agent.flatten_action(action_tuple[0], action_tuple[1])
            next_state, reward, done, info = env.step(action)

            agent.store_transition((state, action, reward, next_state, float(done)))
            state = next_state
            ep_reward += reward
            agent.train(batch_size)
        all_rewards.append(ep_reward)
        if ep % target_update_freq == 0:
            agent.update_target_network()
        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)
        print(f"Episode {ep + 1}/{episodes}: Total Reward: {ep_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
    env.close()
    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())
    #print("----- Session info saved at:", path)
    return agent, all_rewards


#################################
# Main Execution: Train & Test  #
#################################
if __name__ == "__main__":
    # Use SUMO GUI mode so you can visually inspect the simulation.
    sumo_cmd = ["sumo", "-c", "one_intersection/simple_intersection.sumocfg"]

    trained_agent, rewards = train_dqn(sumo_cmd, episodes=500, batch_size=64)

    # Save the trained model parameters
    save_path=f"models/dqn_fourway_model.pth"
    torch.save(trained_agent.q_network.state_dict(), save_path)
    print("Model saved.")
    # Visualization code (from your visualization module)
    viz = Visualization()
    viz.save_data_and_plot(data=rewards, filename='Training Rewards', xlabel='Episode', ylabel='Total Reward')


