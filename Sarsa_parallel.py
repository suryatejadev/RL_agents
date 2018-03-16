import numpy as np
from MountainCar import *
from multiprocessing import Pool
import itertools

class Agent_SARSA:
    def __init__(self,alpha=0.05,gamma=1,epsilon=0.5,order_fourier=1,state_dim=2):
        # Define actions
        self.action = np.array(([1,0,-1]))
        # Set hyperparameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.order_fourier = order_fourier
        # Set Fourier Transform parameters
        self.n = (order_fourier+1)**state_dim
        self.m = self.action.size
        iter = itertools.product(range(self.order_fourier+1),repeat=state_dim)
        self.fourier_basis = np.array([list(map(int,x)) for x in iter])

    def init_weights(self):
        self.weights = np.zeros((self.m,self.n))

    def get_action(self,state=None):

        # Get fourier transform of state
        feat_vector = np.zeros((self.n))
        for i in range(self.n):
            feat_vector[i] = np.cos(np.pi*np.dot(self.fourier_basis[i],state))
        # Get state-action value function. The list corresponds to
        # the q values for the given state and all possible actions
        q_func = np.dot(self.weights,feat_vector[:,np.newaxis]).T[0]
        # with prob eps, choose action according to a uniform distribution
        if np.random.rand(1)[0]<self.epsilon:
            action_index = np.random.randint(self.m)
            fourier_cache = [feat_vector,q_func[action_index],action_index]
            return self.action[action_index],fourier_cache
        # with prob 1-eps, choose action, randomly among all possible actions,
        # that gives maximum q_func value
        else:
            loc_max = np.where(q_func==np.array((q_func.max())))[0]
            action_index = loc_max[np.random.choice(len(loc_max))]
            fourier_cache = [feat_vector,q_func[action_index],action_index]
            return self.action[action_index],fourier_cache

    # Update the weights using SARSA
    def update(self,present_val,reward,next_val=None):
        # If next state is terminal state, then its value is 0
        if next_val==None:
            td_error = reward - present_val[1]
        else:
            td_error = reward + self.gamma*next_val[1] - present_val[1]
        self.weights[present_val[2],:] += self.alpha * td_error * present_val[0]

def Run_Sarsa_trial(parameters_index):

    parameters = parameters_index[1]
    print parameters_index[0]
    num_episodes = parameters['num_episodes']
    num_timesteps = parameters['num_timesteps']

    alpha = parameters['alpha']
    gamma = parameters['gamma']
    epsilon = parameters['epsilon']
    order_fourier = parameters['order_fourier']

    # Initialize environment and agent
    env = env_MountainCar()

    agent = Agent_SARSA(alpha=alpha, gamma=gamma,
             epsilon=epsilon, order_fourier=order_fourier,
             state_dim = env.state_dim)

    # Initialize undiscounted returns
    undiscounted_returns = np.zeros((num_episodes))

    # Initialize weights to 0
    agent.init_weights()
    #Each iteration is one run of an episode
    for iter_episode in range(num_episodes):

        # Sample initial states (s)
        present_state = env.init_states()
        if env.check_terminality(present_state)==True:
            continue

        # Each iteration is one time step in an episode
        for iter_time in range(num_timesteps):

            # Choose an action using eps-greedy (a)
            present_action,present_cache = agent.get_action(state=env.normalize(present_state))
            # Find the next state (s')
            next_state = env.get_nextState(state= present_state,action = present_action)
            # Find the reward (r)
            reward = env.get_reward(state = next_state)
            # Accumulate the reward in 'undiscounted_rewards'
            undiscounted_returns[iter_episode] += reward
            # Check if next state is terminal
            if env.terminal_state==True:
                # Update weights of the agent
                agent.update(present_val = present_cache,reward = reward)
                # Reset terminal_state flag
                env.terminal_state = False
                # break from the loop, ending the episode
                break
            else:
                # Choose the next action using eps-greedy (a')
                next_action,next_cache = agent.get_action(state=env.normalize(next_state))
                # Update weights of the agent
                agent.update(present_val = present_cache,reward = reward,next_val = next_cache)
                # Set the  next state to the present state of the agent
                present_state = next_state

    return undiscounted_returns

def Run_Sarsa_parallel(parameters):

    num_trials = parameters['num_trials']
    num_episodes = parameters['num_episodes']
    returns_list = []
    p = Pool()
    returns_list.append(p.map(Run_Sarsa_trial,itertools.izip(range(num_trials),itertools.repeat(parameters))))
    returns = np.zeros((num_trials,num_episodes))
    for i in range(num_trials):
        returns[i,:] = returns_list[0][i]
    return returns



