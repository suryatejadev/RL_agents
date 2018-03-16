import numpy as np

class env_MountainCar:
    def __init__(self):
        self.state_resolution = 1000
        self.state_dim = 2 # position and velocity
        self.terminal_state = False
        # Define possible state values
        self.min_position = -1.2
        self.max_position = 0.5
        self.min_velocity = -0.07
        self.max_velocity = 0.07
        self.position = -0.5
        self.velocity = 0

    def init_states(self):
        return [-0.5,0]

    # Normalize states to [0,1] for fourier basis transform
    def normalize(self,states):
        norm_position = (states[0]-self.min_position)/(self.max_position-self.min_position)
        norm_velocity = (states[1]-self.min_velocity)/(self.max_velocity-self.min_velocity)
        return np.array(([norm_position,norm_velocity]))

    # Return the next state, given the present state and action
    def get_nextState(self,state,action):
        # Find next velocity
        next_velocity = state[1] + 0.001*action - 0.0025*np.cos(3*state[0])
        # Bound the velocity to its limits
        next_velocity = np.clip(next_velocity,self.min_velocity,self.max_velocity)
        # Find next position
        next_position = state[0] + next_velocity
        # Bound the position to its limits
        if next_position>=0.5:
            self.terminal_state = True
        elif next_position<=-1.2:
            next_position = -1.2
            next_velocity = 0
        return np.array(([next_position,next_velocity]))

    def get_reward(self,state):
        if self.terminal_state == True:
            return 0
        else:
            return -1

    def check_terminality(self,state):
        if state[0] >= self.max_position:
            return True
        else:
            return False


