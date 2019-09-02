import numpy as np
import random
from collections import namedtuple, deque
from drqn.drqnarchitecture import DRQNNetwork
import torch
import torch.nn.functional as F
import torch.optim as optim
import itertools

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network
HIDDEN_DIM = 32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DRQNAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed = None):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        if not seed is None:
            random.seed(seed)

        # Q-Network
        self.drqn_behaviour = DRQNNetwork(state_size, action_size, HIDDEN_DIM, 1, seed = seed).to(device)
        self.drqn_target = DRQNNetwork(state_size, action_size, HIDDEN_DIM, 1, seed = seed).to(device)
        self.optimizer = optim.Adam(self.drqn_behaviour.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed = seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        self.flag = 0

    def init_hidden(self):
        weight = next(self.drqn_behaviour.parameters()).data
        # hidden = (weight.new(self.drqn_behaviour.no_layers, BATCH_SIZE, self.drqn_behaviour.hidden_dim).zero_().to(device),
        #               weight.new(self.drqn_behaviour.no_layers, BATCH_SIZE, self.drqn_behaviour.hidden_dim).zero_().to(device))
        hidden = (weight.new(self.drqn_behaviour.no_layers, 1, self.drqn_behaviour.hidden_dim).zero_().to(device),
                      weight.new(self.drqn_behaviour.no_layers, 1, self.drqn_behaviour.hidden_dim).zero_().to(device))
        return hidden

    def step(self, new_episode, state, hidden, action, reward, next_state, next_hidden, done):
        # Save experience in replay memory

        hidden_state, cell_state = hidden
        next_hidden_state, next_cell_state = next_hidden
        self.memory.add(new_episode, state, hidden_state, cell_state, action, reward, next_state, next_hidden_state, next_cell_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
                # sampled_experiences = 0
                # while sampled_experiences < BATCH_SIZE:
                #     experiences = self.memory.sample()
                #     self.learn(experiences, GAMMA)
                #     sampled_experiences += len(experiences)
                # for experience in experiences:
                    # print(e.hidden_state.shape)


    def act(self, state, hidden, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        # state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # state = state.float().unsqueeze(0).to(device)

        self.drqn_behaviour.eval()
        with torch.no_grad():
            action_values, hidden = self.drqn_behaviour(state, hidden)
        self.drqn_behaviour.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()), hidden
        else:
            return random.choice(np.arange(self.action_size)), hidden

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, hidden_states, cell_states, actions, rewards, next_states, next_hidden_states, next_cell_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        # print('next_states: ', next_states.shape)
        # print('next_hidden_states: ',next_hidden_states.shape)
        Q_targets_next = self.drqn_target(next_states, (next_hidden_states, next_cell_states))[0].detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from behaviour model
        Q_expected, hidden = self.drqn_behaviour(states, (hidden_states, cell_states))
        Q_expected = Q_expected.gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.drqn_behaviour, self.drqn_target, TAU)

    def soft_update(self, behaviour_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_behaviour + (1 - τ)*θ_target
        Params
        ======
            behaviour_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, behaviour_param in zip(target_model.parameters(), behaviour_model.parameters()):
            target_param.data.copy_(tau*behaviour_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed = None):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        # self.action_size = action_size
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=buffer_size)
        self.episode_start_indexes = []
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "hidden_state", "cell_state","action", "reward", "next_state", "next_hidden_state", "next_cell_state", "done"])
        if not seed is None:
            self.seed = random.seed(seed)

    def add(self, new_episode, state, hidden_state, cell_state, action, reward, next_state, next_hidden_state, next_cell_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, hidden_state, cell_state, action, reward, next_state, next_hidden_state, next_cell_state, done)
        if len(self.memory)>=self.buffer_size:
            self.episode_start_indexes = [elt - 1 for elt in self.episode_start_indexes]
            if self.episode_start_indexes[0] < 0:
                self.episode_start_indexes = self.episode_start_indexes[1:]
        self.memory.append(e)
        if new_episode:
            self.episode_start_indexes.append(len(self.memory)-1)


    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        # experiences = random.sample(self.memory, k=self.batch_size)
        experiences = []
        # while len(experiences) < self.batch_size:
        #     rand_episode = random.randint(0,len(self.episode_start_indexes)-1)
        #     if rand_episode == len(self.episode_start_indexes) - 1:
        #         new_experiences = list(itertools.islice(self.memory, self.episode_start_indexes[rand_episode], len(self.memory)))
        #     else:
        #         new_experiences = list(itertools.islice(self.memory, self.episode_start_indexes[rand_episode], self.episode_start_indexes[rand_episode+1]))
        #     experiences = experiences + new_experiences


        rand_episode = random.randint(0,len(self.episode_start_indexes)-1)
        if rand_episode == len(self.episode_start_indexes) - 1:
            experiences = list(itertools.islice(self.memory, self.episode_start_indexes[rand_episode], len(self.memory)))
        else:
            experiences = list(itertools.islice(self.memory, self.episode_start_indexes[rand_episode], self.episode_start_indexes[rand_episode+1]))

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        hidden_states = torch.from_numpy(np.vstack([e.hidden_state for e in experiences if e is not None])).float().to(device).transpose(0,1)
        cell_states = torch.from_numpy(np.vstack([e.cell_state for e in experiences if e is not None])).float().to(device).transpose(0,1)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        next_hidden_states = torch.from_numpy(np.vstack([e.next_hidden_state for e in experiences if e is not None])).float().to(device).transpose(0,1)
        next_cell_states = torch.from_numpy(np.vstack([e.next_cell_state for e in experiences if e is not None])).float().to(device).transpose(0,1)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, hidden_states, cell_states, actions, rewards, next_states, next_hidden_states, next_cell_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
