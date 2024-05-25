import pandas as pd
import matplotlib.pyplot as plt
import os

# Load data from CSV file
data = pd.read_csv('./process_data.csv')

# Convert 'Time' column to datetime
data['Time'] = pd.to_datetime(data['Time'])

# Set Time as the index of the dataframe
data.set_index('Time', inplace=True)

# Create a directory to store plots if it doesn't exist
output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)

# Loop through each unique PID and plot its CPU and Memory usage
for pid in data['PID'].unique():
    subset = data[data['PID'] == pid]
    process_name = subset['Process Name'].iloc[0].replace('.exe', '')  # Clean process name

    # Creating a figure with two subplots
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

    # Set a comprehensive title including the process name and PID
    fig.suptitle(f'Process: {process_name}, PID: {pid}', fontsize=16)

    # Plot CPU Usage
    axes[0].plot(subset.index, subset['CPU Usage (%)'], label='CPU Usage (%)', color='b')
    axes[0].set_xlabel('Time')  # X-axis label
    axes[0].set_ylabel('CPU Usage (%)')  # Y-axis label
    axes[0].set_title('CPU Usage Over Time')
    axes[0].legend()

    # Plot Memory Usage
    axes[1].plot(subset.index, subset['Memory Usage (%)'], label='Memory Usage (%)', color='r')
    axes[1].set_xlabel('Time')  # X-axis label
    axes[1].set_ylabel('Memory Usage (%)')  # Y-axis label
    axes[1].set_title('Memory Usage Over Time')
    axes[1].legend()

    # Improve layout to avoid overlap
    plt.tight_layout(rect=[0, 0, 1, 0.95])


#################################################################
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Read the CSV file
input_csv_path = 'network_data.csv'
data = pd.read_csv(input_csv_path)

# Convert 'Time' column to datetime format
data['Time'] = pd.to_datetime(data['Time'])

# Plot bytes sent vs. time
plt.figure(figsize=(10, 5))
plt.plot(data['Time'], data['bytesent'], label='Bytes Sent', color='blue')
plt.xlabel('Time')
plt.ylabel('Bytes Sent')
plt.title('Bytes Sent over Time')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.xticks(rotation=45)
plt.tight_layout()
plt.legend()
plt.show()

# Plot bytes received vs. time
plt.figure(figsize=(10, 5))
plt.plot(data['Time'], data['byterecv'], label='Bytes Received', color='green')
plt.xlabel('Time')
plt.ylabel('Bytes Received')
plt.title('Bytes Received over Time')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.xticks(rotation=45)
plt.tight_layout()
plt.legend()
plt.show()

