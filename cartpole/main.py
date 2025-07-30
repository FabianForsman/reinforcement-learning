from src.environment import CartPoleEnv
from src.renderer import CartPoleRenderer
from agent.q_learning import QLearningAgent

# Number of episodes to train the agent
n_episodes = 500
# Maximum steps per episode
max_steps = 200

# Initialize the CartPole environment
env = CartPoleEnv()
# Initialize the renderer for visualization
renderer = CartPoleRenderer()
# Initialize the Q-learning agent with discretized state bins and number of actions
agent = QLearningAgent(state_bins=[6, 6, 6, 6], n_actions=2)

# Training loop over episodes
for ep in range(n_episodes):
    # Reset the environment at the start of each episode
    state = env.reset()
    total_reward = 0
    # Step through the environment
    for step in range(max_steps):
        # Agent selects an action based on the current state
        action = agent.act(state)
        # Environment returns next state, reward, and done flag
        next_state, reward, done = env.step(action)
        # Agent updates its Q-table based on the transition
        agent.update(state, action, reward, next_state, done)
        # Move to the next state
        state = next_state
        # Accumulate the reward
        total_reward += reward
        # Render the current state (uncomment to visualize)
        renderer.render(state)
        # If the episode is done, exit the loop
        if done:
            break
    # Print the total reward for the episode
    print(f"Episode {ep + 1}, Total Reward: {total_reward}")
