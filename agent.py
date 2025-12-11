
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
#####################
# Q-Network & Agent #
#####################
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)


class DQNAgent:
    def __init__(self, state_dim, action_n, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.991, epsilon_min=0.01):
        self.state_dim = state_dim
        self.action_dim = action_n  # Number of phase actions (e.g., 3)
        #self.duration_n = duration_n  # Number of duration options (e.g., 3)
        #self.composite_action_dim = phase_n * duration_n  # Total discrete actions (9)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = QNetwork(state_dim, self.action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, self.action_dim).to(self.device)
        self.update_target_network()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=5000)

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    # def flatten_action(self, phase_action, duration_action):
    #     # Map tuple (phase_action, duration_action) to single integer index.
    #     return phase_action * self.duration_n + duration_action

    # def unflatten_action(self, index):
    #     phase_action = index // self.duration_n
    #     duration_action = index % self.duration_n
    #     return (phase_action, duration_action)

    def act(self, state):
        if random.random() < self.epsilon:
            action = random.randrange(self.action_dim)
            # duration_action = random.randrange(self.duration_n)
            return action
        state_tensor = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        action = int(torch.argmax(q_values).item())
        return action

    def store_transition(self, transition):
        self.memory.append(transition)

    def sample_memory(self, batch_size):
        return random.sample(self.memory, batch_size)

    def train(self, batch_size):
        if len(self.memory) < batch_size:
            return
        transitions = self.sample_memory(batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        current_q = self.q_network(states).gather(1, actions)
        with torch.no_grad():
            max_next_q = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (self.gamma * max_next_q * (1 - dones))
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

