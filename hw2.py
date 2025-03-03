import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from homework2 import Hw2Env
from matplotlib import pyplot as plt
from matplotlib.pyplot import ion, step

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

class DQN_high(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN_high, self).__init__()
        self.fc1 = nn.Linear(input_shape, 64)
        self.fc2 = nn.Linear(64, n_actions)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
class DQN_Agent():
    def __init__(self, input_shape, n_actions, lr=1e-1, gamma=0.99, batch_size=64, buffer_size=100000, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995, device=torch.device("cpu")):
        
        self.device = device
        self.n_actions = n_actions
        self.policy_net = DQN_high(input_shape, n_actions).to(device)
        self.target_net = DQN_high(input_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()
        self.memory = ReplayMemory(100000)
        self.steps_done = 0
        self.BATCH_SIZE = 64
        self.GAMMA = 0.999
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.TARGET_UPDATE = 10
        self.episode_durations = []
        self.TAU = 0.005

    def select_action(self,state):
        global steps_done
        sample = random.random()
        self.steps_done += 1
        if sample > self.epsilon:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            
            return torch.tensor([[np.random.randint(self.n_actions)]], device=self.device, dtype=torch.long)


    


    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

def running_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def train():
    env = Hw2Env(n_actions=8, render_mode="offscreen")
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    agent = DQN_Agent(6, 8,device=device)

    rewards = []
    rewards_per_step = []
    running_avg_rewards = []
    steps = 0
    target_update_frequency = 200
    update_frequency = 10
    window_size = 100

    plt.ion()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    line1, = ax1.plot(rewards)
    line2, = ax2.plot(rewards_per_step)
    line3, = ax3.plot(running_avg_rewards)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Total Reward per Episode')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Reward')
    ax2.set_title('Reward per Step')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Running Avg Reward')
    ax3.set_title('Running Average of Total Rewards')


    for i_episode in range(10000):
        # Initialize the environment and get its state
        env.reset()
        action = np.random.randint(agent.n_actions)
        state, _, _, _ = env.step(action)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        total_reward = 0
        while not done:
            action = agent.select_action(state)
            observation, reward, terminated, truncated = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            agent.memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            if steps % update_frequency == 0:
                agent.optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            if steps % target_update_frequency == 0:
                target_net_state_dict = agent.target_net.state_dict()
                policy_net_state_dict = agent.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*agent.TAU + target_net_state_dict[key]*(1-agent.TAU)
                agent.target_net.load_state_dict(target_net_state_dict)
            steps += 1
            rewards_per_step.append(reward.item())
            total_reward += reward.item()
        rewards.append(total_reward)
        print(f"Episode {i_episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")

        # Calculate running average of total rewards
        if len(rewards) >= window_size:
            running_avg_rewards.append(np.mean(rewards[-window_size:]))
        
        agent.epsilon = max(agent.epsilon_min, agent.epsilon*agent.epsilon_decay)
        # Update the plots

        ax1.set_ylim(0, max(rewards)*1.1)
        ax2.set_ylim(0, max(rewards_per_step)*1.1)
        if len(running_avg_rewards) > 0:
            ax3.set_ylim(0, max(running_avg_rewards)*1.1)

        line1.set_ydata(rewards)
        line1.set_xdata(range(len(rewards)))
        ax1.relim()
        ax1.autoscale_view()
        
        line2.set_ydata(rewards_per_step)
        line2.set_xdata(range(len(rewards_per_step)))
        ax2.relim()
        ax2.autoscale_view()
        
        line3.set_ydata(running_avg_rewards)
        line3.set_xdata(range(len(running_avg_rewards)))
        ax3.relim()
        ax3.autoscale_view()
        
        plt.draw()
        plt.pause(0.01)
        if i_episode % 1000 == 0:
            torch.jit.save(torch.jit.script(agent.policy_net), "hw2_2_{}_{}.pt".format(i_episode,1e-1))

    # Save the final plots
    plt.ioff()
    plt.savefig("rewards.png")
    print('Complete')

def test():
    env = Hw2Env(n_actions=8, render_mode="gui")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    agent = DQN_Agent(6, 8,device=device, epsilon=0)
    agent.policy_net = torch.jit.load("hw2_26000_0.001.pt").to(device)
    agent.policy_net.eval()
    rewards = []
    for i_episode in range(100):
        env.reset()
        action = np.random.randint(agent.n_actions)
        state, _, _, _ = env.step(action)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        total_reward = 0
        while not done:
            action = agent.select_action(state)
            observation, reward, terminated, truncated = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Move to the next state
            state = next_state
            total_reward += reward.item()
        rewards.append(total_reward)
        print(f"Episode {i_episode}, Total Reward: {total_reward}")
    print('Complete')

if __name__ == "__main__":
    #test()
    train()
