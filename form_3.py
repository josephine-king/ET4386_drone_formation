import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def load_and_initialize(file_path, num_agents, connectivity, control_gain, dt, total_time):
    # Load desired positions from the data file
    data = loadmat(file_path)
    try:
        z_star = data['z_star'].reshape(num_agents, 2)
    except KeyError:
        raise KeyError("Desired positions 'z_star' not found in the data.")
    
    # Initialize adjacency matrix based on connectivity
    if connectivity == 'all':
        adjacency = np.ones((num_agents, num_agents)) - np.eye(num_agents)
    else:
        adjacency = np.zeros((num_agents, num_agents))
        if connectivity == 'nearest':
            for i in range(num_agents):
                adjacency[i][(i + 1) % num_agents] = 1
                adjacency[i][(i - 1) % num_agents] = 1
        elif connectivity == 'project':
            connections = [
                (0,1),(0,2),(0,3),(0,4),
                (1,3),(1,6),
                (2,4),(2,5),
                (3,5),(3,4),
                (4,6),(5,6),
            ]
            for i, j in connections:
                adjacency[i, j] = 1
                adjacency[j, i] = 1
        else:
            raise ValueError("Unsupported connectivity type. Choose 'all', 'nearest', or 'project'.")
    
    # Debug: Print adjacency matrix
    print("Adjacency Matrix:")
    print(adjacency)
    
    # Compute desired relative positions between connected agents
    desired_offsets = {}
    for i in range(num_agents):
        for j in range(num_agents):
            if adjacency[i, j]:
                desired_offsets[(i, j)] = z_star[i] - z_star[j]
    
    positions = np.random.rand(num_agents, 2) * 10  # Random initial positions
    steps = int(total_time / dt)
    traces = [np.empty((0, 2)) for _ in range(num_agents)]
    mse_history = []  # Initialize MSE history list
    
    return desired_offsets, positions, traces, steps, control_gain, adjacency, mse_history

def compute_control(positions, desired_offsets, adjacency, gain):
    NOISE_STD = 0.5  # Standard deviation of the Gaussian noise
    control_inputs = np.zeros_like(positions)
    num_agents = len(positions)
    for i in range(num_agents):
        for j in range(num_agents):
            if adjacency[i, j]:
                # Compute the relative position error
                rel_pos_error = (positions[i] - positions[j]) - desired_offsets[(i, j)]
                noise = np.random.normal(0, NOISE_STD, size=2)
                rel_pos_error_noisy = rel_pos_error + noise
                # Update control input based on the relative position error
                control_inputs[i] -= gain * rel_pos_error_noisy
    return control_inputs

def compute_mse(positions, desired_offsets, adjacency):
    squared_errors = []
    for (i, j), offset in desired_offsets.items():
        rel_pos_error = (positions[i] - positions[j]) - offset
        squared_errors.append(np.linalg.norm(rel_pos_error)**2)
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
    # === Configuration ===
    DATA_FILE = 'data.mat'
    NUM_AGENTS = 7
    DT = 0.01
    TOTAL_TIME = 3  # seconds
    CONTROL_GAIN = 1.0
    CONNECTIVITY = 'project'  # Options: 'all', 'nearest', 'project'

    # === Load Data and Initialize ===
    try:
        desired_offsets, positions, traces, steps, gain, adjacency, mse_history = load_and_initialize(
            DATA_FILE, NUM_AGENTS, CONNECTIVITY, CONTROL_GAIN, DT, TOTAL_TIME
        )
    except Exception as e:
        print(f"Initialization Error: {e}")
        exit(1)

    # === Setup Plot ===
    fig1, ax1, scatter, lines, connection_lines, fig2, ax2, mse_line, mse_data_x, mse_data_y = setup_plot(NUM_AGENTS, adjacency)
    scatter.set_offsets(positions)

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

    # === Simulation Loop ===
    for step in range(steps):
        if not running:
            print("Simulation terminated by user.")
            break

        current_time = step * DT

        # Compute control inputs
        control_inputs = compute_control(positions, desired_offsets, adjacency, gain)

        # Update positions
        positions += control_inputs * DT

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
        mse = compute_mse(positions, desired_offsets, adjacency)
        mse_history.append(mse)
        mse_data_x.append(current_time)
        mse_data_y.append(mse)

        # Update MSE plot data
        mse_line.set_data(mse_data_x, mse_data_y)
        
        # Adjust MSE plot limits dynamically if necessary
        if current_time > ax2.get_xlim()[1]:
            ax2.set_xlim(0, current_time + TOTAL_TIME * 0.1)  # Extend X-axis by 10% of TOTAL_TIME
            ax2.figure.canvas.draw()
        if mse > ax2.get_ylim()[1]:
            # ax2.set_ylim(0, mse * 1.1)  # Extend Y-axis to 10% above current MSE
            ax2.figure.canvas.draw()

        # Redraw the plots
        fig1.canvas.draw()
        fig2.canvas.draw()
        plt.pause(0.001)

    # === Final Visualization ===
    plt.show()

if __name__ == "__main__":
    main()
