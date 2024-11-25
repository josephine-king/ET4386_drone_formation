import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def load_data(file_path):
    """
    Load MATLAB data from a .mat file.

    Parameters:
        file_path (str): Path to the .mat file.

    Returns:
        dict: Dictionary containing loaded MATLAB variables.
    """
    try:
        data = loadmat(file_path)
        return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        exit(1)
    except Exception as e:
        print(f"An error occurred while loading {file_path}: {e}")
        exit(1)

def initialize_positions(num_agents, scale=10):
    """
    Initialize agent positions randomly within a specified scale.

    Parameters:
        num_agents (int): Number of agents.
        scale (float): Scaling factor for initial positions.

    Returns:
        np.ndarray: Array of shape (num_agents, 2) with initial positions.
    """
    return np.random.rand(num_agents, 2) * scale

def compute_laplacian(desired_positions, connectivity='all'):
    """
    Compute the Laplacian matrix based on desired positions and connectivity.

    Parameters:
        desired_positions (np.ndarray): Desired absolute positions for formation.
        connectivity (str): Type of connectivity ('all', 'nearest', etc.).

    Returns:
        np.ndarray: Laplacian matrix of shape (num_agents, num_agents).
    """
    num_agents = desired_positions.shape[0]
    adjacency = np.zeros((num_agents, num_agents))

    if connectivity == 'all':
        # All-to-all connectivity (complete graph)
        adjacency = np.ones((num_agents, num_agents)) - np.eye(num_agents)
    elif connectivity == 'nearest':
        # Each agent is connected to its nearest neighbor(s)
        # Example: connect each agent to its immediate next agent (ring topology)
        for i in range(num_agents):
            adjacency[i][(i + 1) % num_agents] = 1
            adjacency[i][(i - 1) % num_agents] = 1
    else:
        # Custom connectivity can be defined here
        raise ValueError("Unsupported connectivity type. Choose 'all' or 'nearest'.")

    # Degree matrix
    degree = np.sum(adjacency, axis=1)
    D = np.diag(degree)

    # Laplacian matrix
    L = D - adjacency
    return L

def compute_control_input(positions, desired_positions, L, control_gain):
    """
    Compute the control input for each agent based on the Laplacian matrix.

    Parameters:
        positions (np.ndarray): Current positions of agents.
        desired_positions (np.ndarray): Desired absolute positions for formation.
        L (np.ndarray): Laplacian matrix.
        control_gain (float): Gain factor for control input.

    Returns:
        np.ndarray: Control inputs for each agent.
    """
    # Compute the error relative to desired positions
    position_errors = positions - desired_positions

    # Control input based on Laplacian
    u = -control_gain * L.dot(position_errors)

    return u

def handle_key_press(event, running_flag):
    """
    Handle key press events to control the simulation.

    Parameters:
        event: Matplotlib key press event.
        running_flag (list): Mutable list acting as a flag for simulation state.
    """
    if event.key == 'escape':
        running_flag[0] = False

def visualize(positions, traces, ax, scatter_plot, lines):
    """
    Update the scatter plot and traces with new positions.

    Parameters:
        positions (np.ndarray): Current positions of agents.
        traces (list of np.ndarray): History of positions for each agent.
        ax (matplotlib.axes.Axes): Matplotlib Axes object for plotting.
        scatter_plot (PathCollection): Scatter plot object for current positions.
        lines (list of Line2D): Line objects for traces.
    """
    # Update scatter plot
    scatter_plot.set_offsets(positions)

    # Update traces
    for idx, line in enumerate(lines):
        line.set_data(traces[idx][:, 0], traces[idx][:, 1])

    # Optionally, adjust plot limits if agents move beyond current view
    # ax.relim()
    # ax.autoscale_view()

    plt.draw()

def main():
    # === Configuration ===
    DATA_FILE = 'data.mat'
    NUM_AGENTS = 7
    DT = 0.01
    TOTAL_TIME = 10  # seconds
    CONTROL_GAIN = 1.0
    CONNECTIVITY = 'nearest'  # 'all' for complete graph, 'nearest' for ring topology

    # === Load Data ===
    data = load_data(DATA_FILE)

    # Extract variables from the loaded MATLAB data
    try:
        z = data['z']
        z_star = data['z_star']
        K = data['K']
        N = data['N']
        L_loaded = data['L']  # Original 'L' from data.mat
    except KeyError as e:
        print(f"Key error: {e} not found in the data.")
        exit(1)

    # Define desired positions (assuming z_star contains desired positions)
    desired_positions = np.array(z_star).reshape(NUM_AGENTS, 2)

    # === Compute Laplacian Matrix ===
    L = compute_laplacian(desired_positions, connectivity=CONNECTIVITY)

    # === Initialize Simulation ===
    positions = initialize_positions(NUM_AGENTS)
    steps = int(TOTAL_TIME / DT)

    # Initialize traces: list of arrays storing history of positions for each agent
    traces = [np.empty((0, 2)) for _ in range(NUM_AGENTS)]

    # Setup visualization
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title('Agent Positions with Traces (Laplacian-Based Control)')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')

    # Initialize scatter plot for current positions
    scatter_plot = ax.scatter(positions[:, 0], positions[:, 1], color='blue', zorder=5)

    # Initialize Line2D objects for traces with distinct colors
    colors = plt.cm.get_cmap('hsv', NUM_AGENTS)
    lines = []
    for idx in range(NUM_AGENTS):
        line, = ax.plot([], [], linestyle='-', linewidth=1, alpha=0.7, color=colors(idx))
        lines.append(line)

    # Simulation control flag
    running_flag = [True]  # Using list for mutable flag

    # Connect key press event to handler
    plt.gcf().canvas.mpl_connect('key_press_event', lambda event: handle_key_press(event, running_flag))

    # === Simulation Loop ===
    for step in range(steps):
        if not running_flag[0]:
            print("Simulation terminated by user.")
            break

        # Compute control inputs using Laplacian-based control law
        control_inputs = compute_control_input(positions, desired_positions, L, CONTROL_GAIN)

        # Update positions
        positions += control_inputs * DT

        # Update traces
        for idx in range(NUM_AGENTS):
            traces[idx] = np.vstack((traces[idx], positions[idx]))

        # Update visualization
        visualize(positions, traces, ax, scatter_plot, lines)

        # Small pause to update the plot
        plt.pause(0.001)  # Adjust as needed for performance

    # Final visualization: ensure all traces are displayed
    plt.show()

if __name__ == "__main__":
    main()
