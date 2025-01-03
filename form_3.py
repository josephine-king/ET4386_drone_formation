import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from Estimator import Estimator, LS_Estimator, BLUE_Estimator, Kalman_Estimator, Average_Estimator
from scipy.spatial import procrustes

# === Global Variables ===
DATA_FILE = 'data.mat'
NUM_AGENTS = 7
NUM_EDGES = 12
NOISE_EN = True
DEBUG_PRINTS = False
T_vals = [5]
PLOT_POSITIONS = True
ERR_THRESHOLD = 0.5
ESTIMATOR = Average_Estimator 
USE_ESTIMATOR = True

def load_and_initialize():
    # Load desired positions from the data file
    data = loadmat(DATA_FILE)
    try:
        desired_positions = data['z_star'].reshape(NUM_AGENTS, 2)
    except KeyError:
        raise KeyError("Desired positions 'z_star' not found in the data.")
    
    noise_cov = data['R'].reshape(2,2)
    weights = data['L'].reshape(NUM_AGENTS, NUM_AGENTS)

    # Initialize adjacency matrix based on connectivity
    adjacency = np.zeros((NUM_AGENTS, NUM_AGENTS))
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

    positions = data['z'].reshape(NUM_AGENTS, 2)  # Random initial positions
    steps = np.ndarray.item(data['K'])
    dt = np.ndarray.item(data['dt'])*10
    traces = [np.empty((0, 2)) for _ in range(NUM_AGENTS)]
    
    return desired_positions, positions, noise_cov, weights, traces, steps, dt, connections, adjacency

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

def compute_control_noise(estimate, connections, weights, noise_cov):
    control_inputs = np.zeros((NUM_AGENTS, 2))
    for i,j in connections:
        noise = np.random.multivariate_normal(np.zeros_like(estimate[i]), noise_cov) if NOISE_EN else 0
        weighted_position_diff = (estimate[i] - estimate[j] + noise) * weights[i, j]
        control_inputs[i] += weighted_position_diff
    return control_inputs

def compute_error(positions, desired_positions):
    squared_errors = []
    for agent in range(NUM_AGENTS):
        error = positions[agent] - desired_positions[agent]
        squared_errors.append(np.linalg.norm(error))
    error = np.sum(squared_errors) if squared_errors else 0
    return error

def update_traces(traces, positions):
    for idx in range(len(traces)):
        traces[idx] = np.vstack((traces[idx], positions[idx]))

def setup_error_plots(steps):
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    if (USE_ESTIMATOR == False):
        ax2.set_title('Error between true and desired positions')
    else:
        ax2.set_title('{name}: Error between true and desired positions'.format(name = ESTIMATOR.__name__))
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Error')
    ax2.set_xlim(0, steps)
    ax2.set_ylim(0, 60)  
    error_data_x = []
    error_data_y = []

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    if (USE_ESTIMATOR == False):
        ax3.set_title('Error between true and estimated positions')
    else:
        ax3.set_title('{name}: Error between true and estimated positions'.format(name = ESTIMATOR.__name__))

    ax3.set_xlabel('Time step')
    ax3.set_ylabel('Error')
    ax3.set_xlim(-5, steps)
    ax3.set_ylim(-.001, 0.18) 
    estimate_error_data_x = []
    estimate_error_data_y = []

    fig3.show()
    fig2.show()

    return fig2, ax2, error_data_x, error_data_y, fig3, ax3, estimate_error_data_x, estimate_error_data_y

def setup_plot(num_agents, adjacency, positions, desired_positions):
    # === Agent Positions Plot ===
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.set_xlim(-3+0.75, 3)
    ax1.set_ylim(-2.2, 1.3)
    ax1.set_title('Drone position trajectories: {name}, T = 5'.format(name = ESTIMATOR.__name__))
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

    fig1.show()
    
    return fig1, ax1, scatter, lines, connection_lines

def main():

    # === Load Data and Initialize ===
    try:
        desired_positions, initial_positions, noise_cov, weights, traces, steps, dt, connections, adjacency = load_and_initialize()
    except Exception as e:
        print(f"Initialization Error: {e}")
        exit(1)

    # === Setup Plots ===
    if (PLOT_POSITIONS == True):
        fig1, ax1, scatter, lines, connection_lines = setup_plot(NUM_AGENTS, adjacency, initial_positions, desired_positions)
    fig2, ax2, error_data_x, error_data_y, fig3, ax3, estimate_error_data_x, estimate_error_data_y = setup_error_plots(steps)
    max_error = 0
    steps_to_err_threshold = []
    total_u_used = []

    # === Simulation Control Flag ===
    running = True
    def on_key(event):
        nonlocal running
        if event.key == 'escape':
            running = False

    if (PLOT_POSITIONS == True):
        fig1.canvas.mpl_connect('key_press_event', on_key)
    fig2.canvas.mpl_connect('key_press_event', on_key)
    fig3.canvas.mpl_connect('key_press_event', on_key)

    for idx in range(0, len(T_vals)):
        
        T = T_vals[idx] 
        error_history = []  # Initialize MSE history list
        estimate_error_history = []
        u_used = 0
        positions = np.matrix.copy(initial_positions)

        # Set up estimator 
        estimator = ESTIMATOR(NUM_AGENTS, NUM_EDGES, connections, noise_cov, initial_positions, T, weights, dt)
        print("Running ", type(estimator).__name__, "with T = ", T)

        # === Simulation Loop ===
        for step in range(steps):
            if not running:
                print("Simulation terminated by user.")
                break
            
            # Get measurements
            if (USE_ESTIMATOR == True):
                measurements_block = get_T_measurements(positions, connections, noise_cov,T)
                estimate = estimator.estimate(measurements_block)
                control_inputs = compute_control(estimate, connections, weights)
                u_used += np.sum(np.abs(control_inputs))
                # Calculate the error between the estimate and the true positions
                _, _, estimate_error = procrustes(estimate, positions)
                estimate_error_history.append(np.sqrt(estimate_error))
            else: 
                if (NOISE_EN == True):
                    control_inputs = compute_control_noise(positions, connections, weights, noise_cov)
                else:
                    control_inputs = compute_control(positions, connections, weights)
                u_used += np.sum(np.abs(control_inputs))

            # Update positions
            for agent in [3,4,5,6]:
                positions[agent] += control_inputs[agent] * dt
            
            if (PLOT_POSITIONS == True):
                # Update traces
                update_traces(traces, positions)

                # Update plot
                scatter.set_offsets(positions)
                for idx, line in enumerate(lines):
                    if traces[idx].size > 0:
                        line.set_data(traces[idx][:, 0], traces[idx][:, 1])
                for connection in connection_lines:
                    line, i, j = connection
                    line.set_data([positions[i, 0], positions[j, 0]], [positions[i, 1], positions[j, 1]])

            # === Error Calculations ===
            error = compute_error(positions, desired_positions)
            error_history.append(error)
            if (error < ERR_THRESHOLD and len(steps_to_err_threshold) <= idx):
                steps_to_err_threshold.append(step)
                    
        if (PLOT_POSITIONS == True):
            ax1.set_aspect('equal', adjustable='datalim')  # Ensure equal scaling
            ax1.figure.canvas.draw()
            fig1.canvas.draw()

        total_u_used.append(u_used)
        max_error = max(max(error_history), max_error)
        ax2.plot(range(steps), error_history, label=f"T = {T}")
        if (USE_ESTIMATOR == True):
            ax3.plot(range(steps), estimate_error_history, label=f"T = {T}")

    # Adjust error plot limits dynamically if necessary
    ax2.set_ylim(0,max_error)
    if step > ax2.get_xlim()[1]:
        ax2.set_xlim(0, step + steps * 0.1)
        ax2.figure.canvas.draw()
    if error > ax2.get_ylim()[1]:
        ax2.figure.canvas.draw()
    ax2.legend()
    fig2.canvas.draw()

    if (USE_ESTIMATOR == True):
        # Adjust error plot limits dynamically if necessary
        if step > ax3.get_xlim()[1]:
            ax3.set_xlim(0, step + steps * 0.1) 
            ax3.figure.canvas.draw()
        if error > ax3.get_ylim()[1]:
            ax3.figure.canvas.draw()
        ax3.legend()
        fig3.canvas.draw()

    print("Time steps to converge to ", ERR_THRESHOLD, " error")
    for idx in range(0, len(T_vals)):
        print("T = ", T_vals[idx], ": ", steps_to_err_threshold[idx])

    print("Total input used: ")
    for idx in range(0, len(T_vals)):
        print("T = ", T_vals[idx], ": ", total_u_used[idx])
    
    # === Final Visualization ===
    plt.show()

if __name__ == "__main__":
    main()


