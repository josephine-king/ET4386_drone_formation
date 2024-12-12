import numpy as np
from scipy.linalg import block_diag
class Estimator:

    def __init__(self, num_agents, num_edges, connections, noise_cov, initial_position, T):
        self.initialized = False
        self.num_agents = num_agents
        self.num_edges = num_edges
        self.connections = connections
        self.noise_cov = noise_cov
        # Retain states and measurements from T time steps
        self.T = T
        # self.past_estimates = [np.reshape(initial_position, (num_agents*2, 1), order="C")]
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

        self.measurements_block = np.zeros((T,num_agents,2))

        self.x_n = np.zeros((self.T,num_agents,2))

    #convert measurements to positions.
    def process_data(self,measurement_block):
        for t in range(self.T):
            measurement = measurement_block[t]
            measurement = np.reshape(measurement, (self.num_edges*2*2, 1), order="C") 
            h_n = np.matmul(self.h_inv, measurement)
            self.x_n[t] = (np.reshape(h_n, (self.num_agents, 2), order="C"))
            
    def estimate(self):
        # average - benchmark
        x_tilda = np.zeros((self.num_agents,2))
        for t in range(self.T):
            x_tilda+=self.x_n[t]
        x_tilda= x_tilda/self.T
        return x_tilda

class LS_Estimator(Estimator):
    def __init__(self, num_agents, num_edges, connections, noise_cov, initial_position, T):
        super().__init__(num_agents, num_edges, connections, noise_cov, initial_position, T) 

    def estimate(self, measurements):
        measurements = np.reshape(measurements, (self.num_edges*2*2*self.T, 1), order="C")
        pseudo_inv_H = np.linalg.pinv(self.H)
        current_estimate = np.matmul(pseudo_inv_H, measurements)
        return (np.reshape(current_estimate, (self.num_agents, 2), order="C"))


# BLUE estimator
class BLUE_Estimator(Estimator):
    def __init__(self, num_agents, num_edges, connections, noise_cov, initial_position, T):
        super().__init__(num_agents, num_edges, connections, noise_cov, initial_position, T)

    def estimate(self, measurements):
        measurements = np.reshape(measurements, (self.num_edges*2*2*self.T, 1), order="C")
        H_trans_C_inv = np.matmul(self.H_trans, self.C_inv)
        est1 = np.linalg.inv(np.matmul(H_trans_C_inv, self.H) + 1e-6*np.identity(self.num_agents*2))
        est2 = np.matmul(est1, H_trans_C_inv)
        current_estimate = np.matmul(est2, measurements)
        return (np.reshape(current_estimate, (self.num_agents, 2), order="C"))

class Kalman(Estimator):
    def __init__(self, num_agents, num_edges, connections, noise_cov, initial_position, T):
        super().__init__(num_agents, num_edges, connections, noise_cov, initial_position, T)
        self.estimators = []
        for state in range(self.num_agents):
            kf = KalmanFilter2D(
                initial_state=initial_position[state,:],
                initial_covariance=noise_cov.copy(),
                process_noise_cov=np.eye(2)*.001,
                measurement_noise_cov=noise_cov
            )
            self.estimators.append(kf)

    def estimate(self,control):
        positions_estimate = np.zeros((self.num_agents,2))
        #predict via latest control
        for agent in range(self.num_agents):
                self.estimators[agent].predict(control[agent])
                self.estimators[agent].update(self.x_n[0][agent])
        #predict over consequent measurments without control
        for t in range(self.T):
            for agent in range(self.num_agents):
                self.estimators[agent].predict((0,0))
                self.estimators[agent].update(self.x_n[t][agent])

        for agent in range(self.num_agents):
            positions_estimate[agent] = self.estimators[agent].get_current_state()
        return positions_estimate

class Kalman_Estimator(Estimator):
    def __init__(self, num_agents, num_edges, connections, noise_cov, initial_position, T, weights, DT):
        super().__init__(num_agents, num_edges, connections, noise_cov, initial_position, T)
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


class KalmanFilter2D:
    def __init__(self, initial_state, initial_covariance, process_noise_cov, measurement_noise_cov):
        """
        Initializes the Kalman Filter with the initial state and covariance matrices.

        Parameters:
        - initial_state: Initial estimate of the state vector (2-element numpy array)
        - initial_covariance: Initial estimate of the state covariance matrix (2x2 numpy array)
        - process_noise_cov: Process noise covariance matrix Q (2x2 numpy array)
        - measurement_noise_cov: Measurement noise covariance matrix R (2x2 numpy array)
        """
        # State vector (position in 2D)
        self.x_hat = initial_state  # Shape: (2,)
        
        # State covariance matrix
        self.P = initial_covariance  # Shape: (2,2)
        
        # Process noise covariance matrix
        self.Q = process_noise_cov  # Shape: (2,2)
        
        # Measurement noise covariance matrix
        self.R = measurement_noise_cov  # Shape: (2,2)
        
        # State transition matrix (identity for position-only model)
        self.F = np.eye(2)
        
        # Control input matrix
        self.B = np.eye(2)
        
        # Measurement matrix (identity, since we measure position directly)
        self.H = np.eye(2)
    
    def predict(self, control_input):
        """
        Performs the prediction step of the Kalman filter.

        Parameters:
        - control_input: Control input vector u_k (2-element numpy array)
        """
        # Predict the state estimate
        self.x_hat = self.F @ self.x_hat + self.B @ control_input
        
        # Predict the estimate covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, measurement):
        """
        Performs the update step of the Kalman filter with the new measurement.

        Parameters:
        - measurement: Measurement vector z_k (2-element numpy array)
        """
        # Compute the Kalman Gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update the state estimate
        y = measurement - self.H @ self.x_hat  # Innovation or measurement residual
        self.x_hat = self.x_hat + K @ y
        
        # Update the estimate covariance
        I = np.eye(2)
        self.P = (I - K @ self.H) @ self.P
    
    def get_current_state(self):
        return self.x_hat
    
    def get_current_covariance(self):
        return self.P