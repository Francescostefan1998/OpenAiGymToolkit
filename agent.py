from collections import defaultdict
import numpy as np

class Agent:
    def __init__(
            self, env,
            learning_rate = 0.01,
            discount_factor = 0.9,
            epsilon_greedy = 0.9,
            epsilon_min = 0.1,
            epsilon_decay = 0.95):
        self.env = env 
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_greedy
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
    
    def choose_action(self, state):
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.env.nA)
        else:
            q_vals = self.q_table[state]
            perm_actions = np.random.permutation(self.env.nA)
            q_vals = [q_vals[a] for a in perm_actions]
            perm_q_argmax = np.argmax(q_vals)
            action = perm_actions[perm_q_argmax]
        return action
 
    def _learn(self, trasition):
        s, a, r, next_s, done = trasition
        q_val = self.q_table[s]