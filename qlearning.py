from gridworld_env import GridWorldEnv
from agent import Agent
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1)   

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

def convert_state_to_hashable(state):
    """Convert state to a hashable format for use as dictionary key"""
    if isinstance(state, dict):
        # Sort by keys to ensure consistent ordering
        return tuple(sorted(state.items()))
    elif isinstance(state, (list, np.ndarray)):
        return tuple(state)
    elif isinstance(state, tuple):
        # Handle tuples that might contain unhashable elements
        converted_elements = []
        for element in state:
            if isinstance(element, dict):
                converted_elements.append(tuple(sorted(element.items())))
            elif isinstance(element, (list, np.ndarray)):
                converted_elements.append(tuple(element))
            else:
                converted_elements.append(element)
        return tuple(converted_elements)
    else:
        # For single values or other types
        return state

def run_qlearning(agent, env, num_episodes=50):
    history = []
    for episode in range(num_episodes):
        state = env.reset()
        state = convert_state_to_hashable(state)
        
        env.render()
        total_reward, n_moves = 0.0, 0
        
        while True:
            action = agent.choose_action(state)
            
            # Fixed: Standard gym environments return 4 values
            try:
                next_s, reward, done, info = env.step(action)
            except ValueError:
                # If your environment returns 5 values, use this instead:
                next_s, reward, done, info, additional_var = env.step(action)
            
            # Convert next_s to hashable format
            next_s = convert_state_to_hashable(next_s)
            
            # Debug print to check state types
            if episode == 0 and n_moves == 0:
                print(f"Debug - State type: {type(state)}, State: {state}")
                print(f"Debug - Next state type: {type(next_s)}, Next state: {next_s}")
            
            agent._learn(Transition(state, action, reward, next_s, done))
            env.render()
            
            state = next_s
            n_moves += 1
            total_reward += reward  # Accumulate total reward
            
            if done:
                break
        
        history.append((n_moves, total_reward))
        print(f'Episode {episode + 1}: Moves = {n_moves}, Total Reward = {total_reward:.2f}')
    
    return history

def plot_learning_history(history):
    fig = plt.figure(1, figsize=(14, 10))
    
    # Plot number of moves
    ax = fig.add_subplot(2, 1, 1)
    episodes = np.arange(len(history))
    moves = np.array([h[0] for h in history])
    plt.plot(episodes, moves, lw=4, marker='o', markersize=10)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.xlabel('Episode', size=20)
    plt.ylabel('# moves', size=20)
    
    # Plot rewards
    ax = fig.add_subplot(2, 1, 2)
    rewards = np.array([h[1] for h in history])
    plt.step(episodes, rewards, lw=4)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.xlabel('Episode', size=20)
    plt.ylabel('Total Rewards', size=20)
    
    plt.tight_layout()  # Added for better spacing
    plt.savefig('qlearning_history.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    env = GridWorldEnv(num_rows=5, num_cols=6)
    agent = Agent(env)
    history = run_qlearning(agent, env)
    env.close()
    plot_learning_history(history)