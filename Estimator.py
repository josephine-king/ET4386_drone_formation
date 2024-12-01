import numpy as np

class Estimator:

    def __init__(self, num_agents, num_edges, connections, noise_cov, initial_position):
        self.initialized = False
        self.num_agents = num_agents
        self.num_edges = num_edges
        self.connections = connections
        self.noise_cov = noise_cov
        self.past_estimate = np.reshape(initial_position, (num_agents*2, 1), order="C")

        # start by obtaining H matrix
        self.H = np.zeros((num_edges*2*2, num_agents*2))
        for idx in range(num_edges*2):
            (i,j) = connections[idx]
            self.H[2*idx,2*i] = 1
            self.H[2*idx,2*j] = -1
            self.H[2*idx+1,2*i+1] = 1
            self.H[2*idx+1,2*j+1] = -1
        print(self.H)
            
    def estimate(self, measurements):
        measurements = np.reshape(measurements, (self.num_edges*2*2, 1), order="C")
        current_estimate = np.matmul(np.linalg.pinv(self.H), measurements)
        past_estimate = current_estimate
        return (np.reshape(current_estimate, (self.num_agents, 2), order="C"))

# Attempt at an SLS Estimator. Currently not working at all
'''    
class SLS_Estimator(Estimator):
    def __init__(self, num_agents, num_edges, connections, noise_cov, initial_position):
        super().__init__(num_agents, num_edges, connections, noise_cov, initial_position) 
        self.C = np.kron(np.identity(self.num_edges*2), 0.05)
        C_inv = np.linalg.inv(self.C)
        self.H_trans = np.transpose(self.H)
        self.past_sigma = np.linalg.inv(np.matmul(np.matmul(self.H_trans, C_inv), self.H))

    def estimate(self, measurements):
        # estimator update
        K_num = np.matmul(self.past_sigma, self.H)
        K_denom = self.C + np.matmul(np.matmul(self.H_trans, self.past_sigma), self.H)
        K = np.matmul(K_num, np.linalg.inv(K_denom))
        current_estimate = self.past_estimate + np.matmul(K, (measurements - np.matmul(np.transpose(H), self.past_estimate)))
        # covariance update
        current_sigma = np.matmul((np.identity(self.num_edges*2)-np.matmul(K, self.H_trans)), self.past_sigma)

        self.past_estimate = current_estimate
        self.past_sigma = current_sigma
        return current_estimate
'''

# BLUE estimator. Not yet implemented
'''
class BLUE_Estimator(Estimator)
    def __init__(self, num_agents, num_edges, connections, noise_cov, initial_position):
        super().__init__(num_agents, num_edges, connections, noise_cov, initial_position) 

    def estimate(self, measurements):
        return current_estimate
'''