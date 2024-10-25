import numpy as np
import matplotlib.pyplot as plt
import math
import time
import matplotlib

# Use TkAgg backend for interactive plotting
matplotlib.use('TkAgg')


def visualize_data(data, anomalies):
    """
    Visualizes the complete data stream along with any detected anomalies.

    Parameters:
        data (np.array): The generated data stream.
        anomalies (list): List of tuples containing anomaly indices and values.
    """
    fig, ax = plt.subplots()
    ax.plot(data, label='Data Stream', color='blue')  # Plot the data stream

    # Mark anomalies in the data
    if anomalies:
        anomaly_idx, anomaly_values = zip(*anomalies)  # Separate anomaly indices and values
        ax.scatter(anomaly_idx, anomaly_values, color='red', label='Anomalies', marker='x')  # Plot anomalies

    ax.legend()
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Data Stream with Anomalies')
    plt.show()


def generate_data_stream(size):
    """
    Generates a synthetic data stream with seasonal patterns, noise, and anomalies.

    Parameters:
        size (int): The number of data points to generate.

    Returns:
        np.array: Array containing the generated data stream.
    """
    time = np.arange(0, size, 1)  # Time points
    seasonal = 10 * np.sin(2 * math.pi * time / 100)  # Seasonal component
    noise = np.random.normal(0, 1, size)  # Random noise
    anomalies = (np.random.rand(size) > 0.98) * 20  # Inject anomalies
    return seasonal + noise + anomalies  # Return combined data stream


def detect_anomalies(data, window_size=20, threshold=3):
    """
    Detects anomalies in a data stream using a sliding window and Z-score method.

    Parameters:
        data (np.array): The data stream.
        window_size (int): The size of the moving window for anomaly detection.
        threshold (float): The Z-score threshold for identifying anomalies.

    Returns:
        list: List of tuples where each tuple contains the index and value of an anomaly.
    """
    anomalies = []  # List to hold detected anomalies
    mean = np.mean(data[:window_size])  # Initial mean
    std = np.std(data[:window_size])  # Initial standard deviation

    # Loop through the data to detect anomalies
    for i in range(window_size, len(data)):
        z_score = (data[i] - mean) / std  # Z-score calculation
        if abs(z_score) > threshold:
            anomalies.append((i, data[i]))  # Flag anomaly if Z-score exceeds threshold

        # Update mean and std for the sliding window
        mean = mean * (window_size - 1) / window_size + data[i] / window_size
        std = np.std(data[i - window_size:i])  # Update std using current window

    return anomalies


def live_visualization(data, window_size=20, threshold=3, refresh_rate=10):
    """
    Performs live visualization of the data stream and detects anomalies in real-time.

    Parameters:
        data (np.array): The data stream.
        window_size (int): The size of the moving window for anomaly detection.
        threshold (float): The Z-score threshold for identifying anomalies.
        refresh_rate (int): The interval to refresh the plot.
    """
    # Variables for storing data points and detected anomalies
    x_data, y_data = [], []
    anomaly_x, anomaly_y = [], []

    # Set up interactive plotting
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2, label='Data Stream', color='blue')  # Line for data stream
    scatter, = ax.plot([], [], 'rx', label='Anomalies')  # Red 'x' for anomalies
    ax.set_ylim(np.min(data) - 5, np.max(data) + 5)  # Set y-axis limits
    ax.legend()

    # Main loop for updating the plot
    for frame in range(len(data)):
        # Append new data point
        x_data.append(frame)
        y_data.append(data[frame])
        line.set_data(x_data, y_data)  # Update line data

        # Detect anomalies in real-time
        if frame >= window_size:
            window = y_data[-window_size:]  # Get current window data
            mean = np.mean(window)  # Calculate mean of the window
            std = np.std(window)  # Calculate std of the window
            z_score = (y_data[-1] - mean) / std  # Calculate Z-score
            if abs(z_score) > threshold:
                # Mark as anomaly
                anomaly_x.append(frame)
                anomaly_y.append(data[frame])
                scatter.set_data(anomaly_x, anomaly_y)  # Update anomaly points

        # Update x-axis to show recent data
        ax.set_xlim(max(0, frame - 100), frame + 10)

        # Refresh the plot with a delay to slow down visualization
        if frame % refresh_rate == 0:
            fig.canvas.draw()  # Draw the updated figure
            fig.canvas.flush_events()  # Flush events to ensure the plot updates
            time.sleep(0.1)  # Increase delay for slower visualization

    plt.ioff()  # Turn off interactive mode
    plt.show()  # Show final plot


# Main function to execute the program
if __name__ == '__main__':
    data_size = 1000  # Total number of data points
    data = generate_data_stream(data_size)  # Generate the data stream

    # Detect anomalies in the full data stream for comparison
    anomalies = detect_anomalies(data, window_size=20, threshold=3)

    # Full data visualization after processing
    visualize_data(data, anomalies)

    # Real-time visualization with anomaly detection
    live_visualization(data, window_size=20, threshold=3,
                       refresh_rate=10)  # Increased refresh_rate for slower animation
