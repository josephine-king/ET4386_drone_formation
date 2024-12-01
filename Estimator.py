import numpy as np

class Estimator:

    def __init__(self, num_agents, num_edges, connections, noise_cov, initial_position, T):
        self.initialized = False
        self.num_agents = num_agents
        self.num_edges = num_edges
        self.connections = connections
        self.noise_cov = noise_cov
        # Retain states and measurements from T time steps
        self.T = T
        self.past_estimates = [np.reshape(initial_position, (num_agents*2, 1), order="C")]
        self.past_measurements = []
        
        # The positions of nodes 1 - 3 are known        
        self.known_nodes = 3
        self.known_positions = np.reshape(initial_position[0:self.known_nodes*2, 0], (1,6))

        # h matrix
        self.h = np.zeros((num_edges*2*2, num_agents*2))
        for idx in range(num_edges*2):
            (i,j) = connections[idx]
            self.h[2*idx,2*i] = 1
            self.h[2*idx,2*j] = -1
            self.h[2*idx+1,2*i+1] = 1
            self.h[2*idx+1,2*j+1] = -1
        self.h_trans = np.transpose(self.h)
        # initialize the H matrix as h
        self.H = self.h

        # noise covariance matrix
        self.C0 = np.kron(np.identity(self.num_edges*2), noise_cov)
        self.C0_inv = np.linalg.inv(self.C0)
        self.C = self.C0
        self.C_inv = self.C0_inv

    def add_measurement(self, measurement):
        measurement = np.reshape(measurement, (self.num_edges*2*2, 1), order="C")
        self.past_measurements.append(measurement)
        if (len(self.past_measurements) > self.T):
            self.past_measurements.pop(0)
        else: 
            self.H = np.kron(np.identity(len(self.past_measurements)), self.h)
            self.H_trans = np.transpose(self.H)
            self.C = np.kron(np.identity(len(self.past_measurements)), self.C0)
            self.C_inv = np.linalg.inv(self.C)
        return measurement

    def add_estimate(self, estimate):
        self.past_estimates.append(estimate)
        if (len(self.past_estimates) > self.T):
            self.past_estimates.pop(0)
            
    def estimate(self, measurement):
        measurement = self.add_measurement(measurement)
        current_estimate = np.matmul(np.linalg.pinv(self.h), measurement)
        self.add_estimate(current_estimate)
        return (np.reshape(current_estimate, (self.num_agents, 2), order="C"))

# Attempt at an SLS Estimator. Currently not working at all
class SLS_Estimator(Estimator):
    def __init__(self, num_agents, num_edges, connections, noise_cov, initial_position, T):
        super().__init__(num_agents, num_edges, connections, noise_cov, initial_position, T) 
        self.sigma = np.linalg.inv(np.matmul(np.matmul(self.h_trans, self.C_inv), self.h))

    def add_measurement(self, measurement):
        measurement = super().add_measurement(measurement)
        sigma = np.linalg.inv(np.matmul(np.matmul(self.H_trans, self.C_inv), self.H))
        return measurement

    def estimate(self, measurement):        
        measurement = self.add_measurement(measurement)
        # estimator update
        K_num = np.matmul(self.sigma, self.h)
        K_denom = self.C0 + np.matmul(np.matmul(self.h_trans, self.sigma), self.h)
        K = np.matmul(K_num, np.linalg.inv(K_denom))
        current_estimate = self.past_estimate + np.matmul(K, (measurement - np.matmul(np.transpose(self.h), self.past_estimates[:])))

        self.add_estimate(current_estimate)
        return (np.reshape(current_estimate, (self.num_agents, 2), order="C"))

# BLUE estimator. Currently does not converge
class BLUE_Estimator(Estimator):
    def __init__(self, num_agents, num_edges, connections, noise_cov, initial_position, T):
        super().__init__(num_agents, num_edges, connections, noise_cov, initial_position, T) 

    def estimate(self, measurement):
        measurement = self.add_measurement(measurement)
        H_trans_C_inv = np.matmul(self.H_trans, self.C_inv)
        factor = np.matmul(np.linalg.inv(np.matmul(H_trans_C_inv, self.H)), H_trans_C_inv)
        current_estimate = np.matmul(factor, np.concatenate(self.past_measurements, axis=0))
        current_estimate = current_estimate[-self.num_agents*2:, 0]
        self.add_estimate(current_estimate)
        return (np.reshape(current_estimate, (self.num_agents, 2), order="C"))
