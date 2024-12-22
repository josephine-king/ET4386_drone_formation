import numpy as np
from scipy.linalg import block_diag
class Estimator:

    def __init__(self, num_agents, num_edges, connections, noise_cov, initial_position, T, weights, DT):
        self.initialized = False
        self.num_agents = num_agents
        self.num_edges = num_edges
        self.connections = connections
        self.noise_cov = noise_cov
        self.T = T
        # The positions of nodes 0 - 2 are known        
        self.known_nodes = [0,1,2]
        self.initial_position = initial_position

        # Calculate the incidence matrix and the system matrix H
        incidence_matrix = np.zeros((num_agents, num_edges*2), dtype=int)
        for edge_index, (u, v) in enumerate(connections):
            incidence_matrix[u][edge_index] = 1
            incidence_matrix[v][edge_index] = -1
        self.H = np.kron(np.transpose(incidence_matrix), np.identity(2))
        self.H_inv = np.linalg.pinv(self.H)
        self.H_trans = np.transpose(self.H)
        
        # Calculate the noise covariance matrix
        self.C = np.kron(np.identity(self.num_edges*2), noise_cov)
        self.C_inv = np.linalg.inv(self.C)

        # Calculate the HT and CT matrices for T measurements
        one_vec = np.ones((self.T,1))
        self.HT = np.kron(one_vec, self.H)
        self.HT_trans = np.transpose(self.HT)
        self.HT_inv = np.linalg.pinv(self.HT)
        self.CT = np.kron(np.identity(T), self.C)
        self.CT_inv = np.linalg.inv(self.CT)

    def estimate(self, measurements):
        pass
        
class MLE_Estimator(Estimator):
    def __init__(self, num_agents, num_edges, connections, noise_cov, initial_position, T, weights, DT):
        super().__init__(num_agents, num_edges, connections, noise_cov, initial_position, T, weights, DT) 
        ones = np.ones((self.T,1))
        self.HT = np.kron(ones.T,self.H_inv)
        
    def estimate(self,measurements):
        measurements = np.reshape(measurements, (self.num_edges*2*2*self.T, 1), order="C")
        
        estimate = np.matmul(self.HT ,measurements)/self.T
        return (np.reshape(estimate, (self.num_agents, 2), order="C"))
    

class LS_Estimator(Estimator):
    def __init__(self, num_agents, num_edges, connections, noise_cov, initial_position, T, weights, DT):
        super().__init__(num_agents, num_edges, connections, noise_cov, initial_position, T, weights, DT) 

    def estimate(self, measurements):
        measurements = np.reshape(measurements, (self.num_edges*2*2*self.T, 1), order="C")
        current_estimate = np.matmul(self.HT_inv, measurements)
        current_estimate = (np.reshape(current_estimate, (self.num_agents, 2), order="C"))
        return current_estimate

# BLUE estimator
class BLUE_Estimator(Estimator):
    def __init__(self, num_agents, num_edges, connections, noise_cov, initial_position, T, weights, DT):
        super().__init__(num_agents, num_edges, connections, noise_cov, initial_position, T, weights, DT)
        HT_trans_CT_inv = np.matmul(self.HT_trans, self.CT_inv)
        denom = np.linalg.inv(np.matmul(HT_trans_CT_inv, self.HT) + 1e-6*np.identity(self.num_agents*2))
        self.BLUE = np.matmul(denom, HT_trans_CT_inv)


    def estimate(self, measurements):
        measurements = np.reshape(measurements, (self.num_edges*2*2*self.T, 1), order="C")
        current_estimate = np.matmul(self.BLUE, measurements)
        current_estimate = (np.reshape(current_estimate, (self.num_agents, 2), order="C"))
        return current_estimate
    

class Kalman_Estimator(Estimator):
    def __init__(self, num_agents, num_edges, connections, noise_cov, initial_position, T, weights, DT):
        super().__init__(num_agents, num_edges, connections, noise_cov, initial_position, T, weights, DT)
        self.initial_position = np.reshape(initial_position, (num_agents * 2, 1), order="C")
        self.state_estimate = np.reshape(initial_position, (num_agents * 2, 1), order="C")
        cov = np.eye(num_agents)
        self.state_cov = np.kron(cov,self.noise_cov)
        self.measurement_cov = self.CT
        self.transition_matrix = np.identity(num_agents*2)
        self.measurement_matrix = self.HT
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
        S = np.matmul(np.matmul(self.HT, self.state_cov), self.HT_trans) + self.measurement_cov  # Innovation covariance
        K = np.matmul(np.matmul(self.state_cov, self.HT_trans), np.linalg.pinv(S))  # Kalman gain

        innovation = measurements - np.matmul(self.HT, predicted_state)
        self.state_estimate = predicted_state + np.matmul(K, innovation)
        self.state_cov = np.matmul(np.identity(self.state_cov.shape[0]) - np.matmul(K, self.HT), self.state_cov)

        self.state_estimate[0:6] = self.initial_position[0:6]

        # Reshape state estimate to return as (num_agents x 2)
        return np.reshape(self.state_estimate, (self.num_agents, 2), order="C")

