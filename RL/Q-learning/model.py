from turtle import forward
from sympy import O
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DeepQNetwork(nn.Module):
    def __init__(self, ALPHA):
        super(DeepQNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(128 * 19 * 8, 512)
        self.fc2 = nn.Linear(512, 6)

        self.optimizer = optim.RMSprop(self.parameters(), lr=ALPHA)
        self.loss = nn.MSELoss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, observation):
        # sourcery skip: inline-immediately-returned-variable
        observation = torch.Tensor(observation).to(self.device)
        observation = observation.unsqueeze(0)
        observation = observation.view(-1, 1, 185, 95)
        observation = F.relu(self.conv1(observation))
        observation = F.relu(self.conv2(observation))
        observation = F.relu(self.conv3(observation))
        observation = observation.view(-1, 128 * 19 * 8)
        observation = F.relu(self.fc1(observation))

        action = self.fc2(observation)

        return action


class Agent(object):
    def __init__(
        self,
        gamma,
        epsilon,
        alpha,
        max_memory_size,
        eps_end=0.05,
        replace=10000,
        action_space=[0, 1, 2, 3, 4, 5],
    ):
        super().__init__()
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.max_memory_size = max_memory_size
        self.eps_end = eps_end
        self.replace_target_cnt = replace
        self.action_space = action_space
        self.steps = 0
        self.learn_step_counter = 0
        self.memory = []
        self.mem_cntr = 0
        self.Q_eval = DeepQNetwork(alpha)
        self.Q_next = DeepQNetwork(alpha)

    def store_transition(self, state, action, reward, state_):
        if self.mem_cntr < self.max_memory_size:
            self.memory.append([state, action, reward, state_])
        else:
            self.memory[self.mem_cntr % self.max_memory_size] = [
                state,
                action,
                reward,
                state_,
            ]
        self.mem_cntr += 1

    def choose_action(self, observation):
        rand = np.random.random()
        action = self.Q_eval.forward(observation=observation)
        if rand < 1 - self.epsilon:
            action = torch.argmax(action[1]).item()
        else:
            action = np.random.choice(self.action_space)
        self.steps += 1
        return action

    def learn(self, batch_size):
        self.Q_eval.optimizer.zero_grad()
        if (
            self.replace_target_cnt is not None
            and self.learn_step_counter % self.replace_target_cnt == 0
        ):
            self.Q_next.load_state_dict(self.Q_eval.state_dict())

        if self.mem_cntr + batch_size < self.max_memory_size:
            mem_start = int(np.random.choice(range(self.mem_cntr)))
        else:
            mem_start = int(
                np.random.choice(range(self.mem_cntr - self.max_memory_size - 1))
            )

        mini_batch = self.memory[mem_start : mem_start + batch_size]
        memory = np.array(mini_batch)

        Q_pred = self.Q_eval.forward(list(memory[:, 0][:])).to(self.Q_eval.device)
        Q_next = self.Q_next.forward(list(memory[:, 3][:])).to(self.Q_eval.device)

        max_action = torch.argmax(Q_next, dim=1).to(self.Q_eval.device)
        rewards = torch.Tensor(list(memory[:, 2])).to(self.Q_eval.device)
        Q_target = Q_pred
        Q_target[:, max_action] = rewards + self.gamma * torch.max(Q_next[1])

        if self.steps > 500:
            if self.epsilon - 1e-4 > self.eps_end:
                self.epsilon -= 1e-4
            else:
                self.epsilon = self.eps_end

        loss = self.Q_eval.loss(Q_target, Q_pred).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.learn_step_counter += 1

