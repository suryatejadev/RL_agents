import numpy as np
from QLearning_parallel import *
from Sarsa_parallel import *
import matplotlib.pyplot as plt
from time import time
from scipy.io import savemat

if __name__=='__main__':

    method = 'sarsa'
    #method = 'QLearning'

    # Trial parameters
    parameters = {}
    parameters['num_trials'] = 10000
    parameters['num_episodes'] = 200
    parameters['num_timesteps'] = 20000

    # Hyper parameters
    parameters['alpha'] = 0.05
    parameters['gamma'] = 1
    parameters['epsilon'] = 0.5
    parameters['order_fourier'] = 1

    # Initialize agent
    if method == 'sarsa':
        t = time()
        returns = Run_Sarsa_parallel(parameters)
        returns_mean,returns_ser = np.mean(returns,axis=0),np.std(returns,axis=0)*1.0/np.sqrt(parameters['num_trials'])
        print time() - t
    if method =='QLearning':
        t = time()
        returns = Run_QLearn_parallel(parameters)
        returns_mean,returns_ser = np.mean(returns,axis=0),np.std(returns,axis=0)*1.0/np.sqrt(parameters['num_trials'])
        print time()-t

    plt.errorbar(np.arange(parameters['num_episodes']),returns_mean,yerr=returns_ser)
    plt.xlabel('Episodes')
    plt.ylabel('Undiscounted Returns')
    plt.title('Episodic Returns over Mountain Car Environment using '+method+' method')
    plt.ylim(-1000,0)
    plt.savefig('plot_'+method+'.png')
    plt.show()

    a = {}
    a['returns'] = returns
    savemat(method+'_return',a)


