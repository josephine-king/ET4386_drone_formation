import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from Estimator import Estimator, LS_Estimator, BLUE_Estimator, Kalman_Estimator

# === Global Variables ===
DATA_FILE = 'data.mat'
NUM_AGENTS = 7
NUM_EDGES = 12
DT = 0.2
TOTAL_TIME = 180  # seconds
NOISE_EN = True
DEBUG_PRINTS = False
T = 1

def load_and_initialize(file_path, num_agents, dt, total_time):
    # Load desired positions from the data file
    data = loadmat(file_path)
    try:
        desired_positions = data['z_star'].reshape(num_agents, 2)
    except KeyError:
        raise KeyError("Desired positions 'z_star' not found in the data.")
    
    noise_cov = data['R'].reshape(2,2)
    weights = data['L'].reshape(num_agents, num_agents)

    # Initialize adjacency matrix based on connectivity
    adjacency = np.zeros((num_agents, num_agents))
    connections = [
        (0,1),(0,2),(0,3),(0,4),
        (1,3),(1,6),
        (2,4),(2,5),
        (3,5),(3,4),
        (4,6),(5,6),
        (1,0),(2,0),(3,0),(4,0),
        (3,1),(6,1),
        (4,2),(5,2),
        (5,3),(4,3),
        (6,4),(6,5),
    ]
    for i, j in connections:
        adjacency[i, j] = 1

    positions = data['z'].reshape(num_agents, 2)  # Random initial positions
    steps = int(total_time / dt)
    traces = [np.empty((0, 2)) for _ in range(num_agents)]
    mse_history = []  # Initialize MSE history list
    
    return desired_positions, positions, noise_cov, weights, traces, steps, connections, adjacency, mse_history

#get raw relative position data.
def get_T_measurements(positions, connections, noise_cov,T):
    measurements = np.zeros((T,len(connections), 2))
    for t in range(len(measurements)):
        for m in range(len(measurements[t])):
            (i,j) = connections[m]
            # Generate noise
            noise = np.random.multivariate_normal(np.zeros_like(positions[i]), noise_cov) if NOISE_EN else 0
            # Compute the position difference between the two agents and add noise
            measurements[t][m] = positions[i] - positions[j] + noise
    return measurements


def compute_control(estimate, connections, weights):
    control_inputs = np.zeros((NUM_AGENTS, 2))
    for i,j in connections:
        weighted_position_diff = (estimate[i] - estimate[j]) * weights[i, j]
        control_inputs[i] += weighted_position_diff
    return control_inputs

def compute_mse(positions, desired_positions):
    squared_errors = []
    for agent in range(NUM_AGENTS):
        error = positions[agent] - desired_positions[agent]
        squared_errors.append(np.linalg.norm(error)**2)
    mse = np.mean(squared_errors) if squared_errors else 0
    return mse

def update_traces(traces, positions):
    for idx in range(len(traces)):
        traces[idx] = np.vstack((traces[idx], positions[idx]))

def setup_plot(num_agents, adjacency):
    # === Agent Positions Plot ===
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_title('Agent Positions with Traces and Connections')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    
    # Setup connection lines first with lower zorder to appear behind traces
    connection_lines = []
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            if adjacency[i, j]:
                line, = ax1.plot([], [], linestyle=':', color='gray', linewidth=1, alpha=0.5, zorder=1)
                connection_lines.append((line, i, j))
    
    # Setup scatter plot and traces with higher zorder
    scatter = ax1.scatter([], [], color='blue', zorder=3)
    colors = plt.cm.get_cmap('hsv', num_agents)
    lines = [ax1.plot([], [], linestyle='-', linewidth=1, alpha=0.7, color=colors(idx), zorder=2)[0] for idx in range(num_agents)]
    
    # === MSE Plot ===
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.set_title('Mean Squared Error (MSE) Over Time')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('MSE')
    mse_line, = ax2.plot([], [], color='red')
    ax2.set_xlim(0, 10)  # Will adjust dynamically
    ax2.set_ylim(0, 100)  # Initial guess; will adjust dynamically
    
    # Initialize MSE data
    mse_data_x = []
    mse_data_y = []
    
    return fig1, ax1, scatter, lines, connection_lines, fig2, ax2, mse_line, mse_data_x, mse_data_y

def main():

    # === Load Data and Initialize ===
    try:
        desired_positions, positions, noise_cov, weights, traces, steps, connections, adjacency, mse_history = load_and_initialize(
            DATA_FILE, NUM_AGENTS, DT, TOTAL_TIME
        )
    except Exception as e:
        print(f"Initialization Error: {e}")
        exit(1)

    # Set up estimator 
    estimator = Kalman_Estimator(NUM_AGENTS, NUM_EDGES, connections, noise_cov, positions, T, weights, DT)
    # # # Create a list to hold the Kalman Filter instances
    # estimators = []

    # # Initialize each estimator with its respective initial state
    # for state in range(NUM_AGENTS):
    #     kf = KalmanFilter2D(
    #         initial_state=positions[state,:],
    #         initial_covariance=noise_cov.copy(),
    #         process_noise_cov=np.eye(2)*.01,
    #         measurement_noise_cov=noise_cov
    #     )
    #     estimators.append(kf)


    # === Setup Plot ===
    fig1, ax1, scatter, lines, connection_lines, fig2, ax2, mse_line, mse_data_x, mse_data_y = setup_plot(NUM_AGENTS, adjacency)
    scatter.set_offsets(positions)
    # Plot desired positions in grey
    ax1.scatter(desired_positions[3:7, 0], desired_positions[3:7, 1], c='grey', s=30, marker='x', label="Desired Positions")

    # Initialize connection lines with initial positions
    for connection in connection_lines:
        line, i, j = connection
        line.set_data([positions[i, 0], positions[j, 0]], [positions[i, 1], positions[j, 1]])

    # Initialize traces lines
    for idx, line in enumerate(lines):
        line.set_data([], [])

    # Initialize MSE plot
    ax2.set_xlim(0, TOTAL_TIME)
    ax2.set_ylim(0, 60)  # Adjust based on expected MSE range
    fig2.show()
    fig1.show()

    # === Simulation Control Flag ===
    running = True
    def on_key(event):
        nonlocal running
        if event.key == 'escape':
            running = False

    fig1.canvas.mpl_connect('key_press_event', on_key)
    fig2.canvas.mpl_connect('key_press_event', on_key)
    positions_estimate = np.zeros((7,2))
    control_inputs = np.zeros((7,2))
    # === Simulation Loop ===
    for step in range(steps):
        if not running:
            print("Simulation terminated by user.")
            break

        current_time = step * DT

        # Get measurements
        measurements_block = get_T_measurements(positions, connections, noise_cov,T)
        estimator.process_data(measurements_block)
        estimate = estimator.estimate(measurements_block)
        control_inputs = compute_control(estimate, connections, weights)

        # Update positions
        for agent in [3,4,5,6]:
            if (DEBUG_PRINTS):
                print(f"Agent: {agent}, old position: {positions[agent]}, new position: {positions_estimate[agent]+control_inputs[agent]}, desired position: {desired_positions[agent]}")
            positions[agent] += control_inputs[agent] * DT


        # Update traces
        update_traces(traces, positions)

        # Update scatter plot
        scatter.set_offsets(positions)


        # Update traces
        for idx, line in enumerate(lines):
            if traces[idx].size > 0:
                line.set_data(traces[idx][:, 0], traces[idx][:, 1])

        # Update connection lines
        for connection in connection_lines:
            line, i, j = connection
            line.set_data([positions[i, 0], positions[j, 0]], [positions[i, 1], positions[j, 1]])

        # === MSE Calculation ===
        mse = compute_mse(positions, desired_positions)
        mse_history.append(mse)
        mse_data_x.append(current_time)
        mse_data_y.append(mse)
        
    max_range = max(np.abs(positions).max(), 1e-3)  # Prevent zero range errors
    ax1.set_xlim(-max_range, max_range+0.5)
    ax1.set_ylim(-max_range, max_range)
    ax1.set_aspect('equal', adjustable='datalim')  # Ensure equal scaling
    ax1.figure.canvas.draw()
    # Redraw the plots
    fig1.canvas.draw()
    
    # plt.pause(0.001)

    # Update MSE plot data
    mse_line.set_data(mse_data_x, mse_data_y)
    
    # Adjust MSE plot limits dynamically if necessary
    ax2.set_ylim(0,5)
    if current_time > ax2.get_xlim()[1]:
        ax2.set_xlim(0, current_time + TOTAL_TIME * 0.1)  # Extend X-axis by 10% of TOTAL_TIME
        ax2.figure.canvas.draw()
    if mse > ax2.get_ylim()[1]:
        # ax2.set_ylim(0, mse * 1.1)  # Extend Y-axis to 10% above current MSE
        ax2.figure.canvas.draw()
    fig2.canvas.draw()
    # === Final Visualization ===
    plt.show()
    print("total mse: ",sum(mse_history))
if __name__ == "__main__":
    main()


