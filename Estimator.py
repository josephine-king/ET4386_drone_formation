import numpy as np
from scipy.linalg import block_diag
class Estimator:

    def __init__(self, num_agents, num_edges, connections, noise_cov, initial_position, T, weights, DT):
        self.initialized = False
        self.num_agents = num_agents
        self.num_edges = num_edges
        self.connections = connections
        self.noise_cov = noise_cov
        # Retain states and measurements from T time steps
        self.T = T
        # self.past_estimates = [np.reshape(initial_position, (num_agents*2, 1), order="C")]
        # The positions of nodes 0 - 2 are known        
        self.known_nodes = [0,1,2]
        self.initial_position = initial_position

        # h matrix
        self.h = np.zeros((num_edges*2*2, num_agents*2))
        for idx in range(num_edges*2):
            (i,j) = connections[idx]
            self.h[2*idx,2*i] = 1
            self.h[2*idx,2*j] = -1
            self.h[2*idx+1,2*i+1] = 1
            self.h[2*idx+1,2*j+1] = -1
        
        self.h_inv = np.linalg.pinv(self.h)
        self.h_trans = np.transpose(self.h)

        one_vec = np.ones((self.T,1))

        # noise covariance matrix
        self.C0 = np.kron(np.identity(self.num_edges*2), noise_cov)
        self.C0_inv = np.linalg.inv(self.C0)

        self.H = np.kron(one_vec, self.h)
        self.H_trans = np.transpose(self.H)
        self.C = np.kron(np.identity(T), self.C0)
        self.C_inv = np.linalg.inv(self.C)
            
    def fill_in_known_nodes(self, estimate):
        for i in self.known_nodes:
            estimate[i*2:i*2+2] = self.initial_position[i*2:i*2+2]
        return estimate

    def estimate(self, measurements):
        pass
        

class LS_Estimator(Estimator):
    def __init__(self, num_agents, num_edges, connections, noise_cov, initial_position, T, weights, DT):
        super().__init__(num_agents, num_edges, connections, noise_cov, initial_position, T, weights, DT) 

    def estimate(self, measurements):
        measurements = np.reshape(measurements, (self.num_edges*2*2*self.T, 1), order="C")
        pseudo_inv_H = np.linalg.pinv(self.H)
        current_estimate = np.matmul(pseudo_inv_H, measurements)
        current_estimate = (np.reshape(current_estimate, (self.num_agents, 2), order="C"))
        current_estimate = self.fill_in_known_nodes(current_estimate)
        return current_estimate

# BLUE estimator
class BLUE_Estimator(Estimator):
    def __init__(self, num_agents, num_edges, connections, noise_cov, initial_position, T, weights, DT):
        super().__init__(num_agents, num_edges, connections, noise_cov, initial_position, T, weights, DT)

    def estimate(self, measurements):
        measurements = np.reshape(measurements, (self.num_edges*2*2*self.T, 1), order="C")
        H_trans_C_inv = np.matmul(self.H_trans, self.C_inv)
        est1 = np.linalg.inv(np.matmul(H_trans_C_inv, self.H) + 1e-6*np.identity(self.num_agents*2))
        est2 = np.matmul(est1, H_trans_C_inv)
        current_estimate = np.matmul(est2, measurements)
        current_estimate = (np.reshape(current_estimate, (self.num_agents, 2), order="C"))
        current_estimate = self.fill_in_known_nodes(current_estimate)
        return current_estimate

class Kalman_Estimator(Estimator):
    def __init__(self, num_agents, num_edges, connections, noise_cov, initial_position, T, weights, DT):
        super().__init__(num_agents, num_edges, connections, noise_cov, initial_position, T, weights, DT)
        self.initial_position = np.reshape(initial_position, (num_agents * 2, 1), order="C")
        self.state_estimate = np.reshape(initial_position, (num_agents * 2, 1), order="C")
        self.state_cov = np.identity(num_agents * 2) * 1e-3
        self.measurement_cov = self.C
        self.transition_matrix = np.identity(num_agents*2)
        self.measurement_matrix = self.H
        self.weights = weights
        self.DT = DT

    def compute_control(self, estimate):
        estimate = np.reshape(estimate, (self.num_agents, 2), order="C")
        control_inputs = np.zeros((self.num_agents, 2))
        for i,j in self.connections:
            weighted_position_diff = (estimate[i] - estimate[j]) * self.weights[i, j]
            control_inputs[i] += weighted_position_diff
        return np.reshape(control_inputs, (self.num_agents*2, 1), order="C")

    def estimate(self, measurements):
        # Reshape measurements
        measurements = np.reshape(measurements, (self.num_edges*2*2*self.T, 1), order="C")

        # Prediction step
        predicted_state = self.state_estimate
        control_inputs = self.compute_control(self.state_estimate)
        predicted_state[6:] += self.DT*control_inputs[6:]

        # Update step
        S = np.matmul(np.matmul(self.H, self.state_cov), self.H_trans) + self.measurement_cov  # Innovation covariance
        K = np.matmul(np.matmul(self.state_cov, self.H_trans), np.linalg.pinv(S))  # Kalman gain

        innovation = measurements - np.matmul(self.H, predicted_state)
        self.state_estimate = predicted_state + np.matmul(K, innovation)
        self.state_cov = np.matmul(np.identity(self.state_cov.shape[0]) - np.matmul(K, self.H), self.state_cov)

        self.state_estimate[0:6] = self.initial_position[0:6]

        # Reshape state estimate to return as (num_agents x 2)
        return np.reshape(self.state_estimate, (self.num_agents, 2), order="C")

