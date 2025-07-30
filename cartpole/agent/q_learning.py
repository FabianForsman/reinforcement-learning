import numpy as np

# Q-learning agent for the CartPole environment.
# This agent discretizes the continuous state space and learns a Q-table
# to select actions using an epsilon-greedy policy.

class QLearningAgent:
    def __init__(self, state_bins, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        """
        Initialize the Q-learning agent.

        Args:
            state_bins (list): Number of bins for discretizing each state dimension.
            n_actions (int): Number of possible actions.
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            epsilon (float): Exploration rate for epsilon-greedy policy.
        """
        self.state_bins = state_bins
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # Q-table shape: (bins for each state dimension..., n_actions)
        self.q_table = np.zeros(state_bins + [n_actions])

    def discretize(self, state):
        """
        Convert a continuous state into discrete bin indices.

        Args:
            state (list or np.array): Continuous state values.

        Returns:
            tuple: Discrete indices for each state dimension.
        """
        # Define upper and lower bounds for each state dimension
        upper_bounds = [2.4, 2.0, np.deg2rad(20), np.deg2rad(50)]
        lower_bounds = [-2.4, -2.0, -np.deg2rad(20), -np.deg2rad(50)]
        # Normalize state values to [0, 1]
        ratios = [
            (s - low) / (high - low)
            for s, low, high in zip(state, lower_bounds, upper_bounds)
        ]
        # Convert normalized values to discrete bin indices
        discretized_state = []
        for i, r in enumerate(ratios):
            bin_index = int(r * self.state_bins[i])
            # Clamp bin_index to valid range
            bin_index = min(self.state_bins[i] - 1, max(0, bin_index))
            discretized_state.append(bin_index)
        return tuple(discretized_state)

    def act(self, state):
        """
        Select an action using epsilon-greedy policy.

        Args:
            state (list or np.array): Current environment state.

        Returns:
            int: Chosen action index.
        """
        d_state = self.discretize(state)
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.choice(self.n_actions)
        # Exploit: best known action
        return np.argmax(self.q_table[d_state])

    def update(self, state, action, reward, next_state, done):
        """
        Update the Q-table using the Bellman equation.

        Args:
            state (list or np.array): State before action.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (list or np.array): State after action.
            done (bool): Whether the episode has ended.
        """
        d_state = self.discretize(state)
        d_next = self.discretize(next_state)
        q_predict = self.q_table[d_state][action]
        # If done, no future reward is considered
        q_target = reward + self.gamma * np.max(self.q_table[d_next]) * (1 - done)
        # Update Q-value towards target
        self.q_table[d_state][action] += self.alpha * (q_target - q_predict)
