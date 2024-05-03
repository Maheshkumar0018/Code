

import pygame
import sys
import psutil
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import defaultdict
import time
import csv
from datetime import datetime
from ui import Checkbox, checkboxes, select_all_checkbox, print_selected_options
import threading
import numpy as np
# import matplotlib.ticker as ticker


# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
WHITE = (255, 255, 255)
FONT_SIZE = 24

# Set up the Pygame window
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Process Monitor")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, FONT_SIZE)

# Function to count the total number of relevant PIDs
def count_pids(process_names):
    pid_cnt = 0
    for process in psutil.process_iter():
        try:
            name = process.name()
            if any(name.lower() in p.lower() for p in process_names):
                pid_cnt += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return pid_cnt

# Function to get process name for a given PID
def get_process_name(pid):
    for process in psutil.process_iter():
        try:
            if process.pid == pid:
                return process.name()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return "Unknown"

# Calculate the total number of relevant PIDs
process_names = ["python.exe"]
total_pids = count_pids(process_names)
print("Total PIDs:", total_pids)

# Create dictionaries to store CPU and memory usage data for each PID
cpu_data = defaultdict(list)
mem_data = defaultdict(list)
time_data = defaultdict(list)

# Function to update CPU and memory usage data for a given process
def update_data(process_name, cpu_data, mem_data, time_data):
    current_time = time.time() # Fetch current time outside the loop
    for process in psutil.process_iter():
        try:
            name = process.name()
            pid = process.pid
            if process_name.lower() in name.lower():
                cpu_percent = process.cpu_percent(interval=None)
                mem_percent = process.memory_percent()
                cpu_data[pid].append(cpu_percent)
                mem_data[pid].append(mem_percent)
                time_data[pid].append(current_time) # Store the timestamp
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

# Function to update CPU and memory usage data using threads
def update_data_threads(process_names, cpu_data, mem_data, time_data):
    threads = []
    for process_name in process_names:
        thread = threading.Thread(target=update_data, args=(process_name, cpu_data, mem_data, time_data))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()


def process_and_reformat_csv(input_csv_path, output_csv_path):
    # Read and process the data
    processed_data = defaultdict(dict)
    with open(input_csv_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        headers = next(reader)  # Skip the header
        for row in reader:
            pid = row[0]
            process_name = row[1]
            timestamp = row[2]
            cpu_usage = float(row[3])
            memory_usage = float(row[4])
            # Assuming you want the last record of a given PID at a given timestamp
            processed_data[timestamp][pid] = (process_name, cpu_usage, memory_usage)

    # Write the reformatted data
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Collect all unique PIDs for header generation
        all_pids = {pid for timestamp in processed_data for pid in processed_data[timestamp]}
        sorted_pids = sorted(all_pids)
        headers = ['Time']
        for pid in sorted_pids:
            headers += [f'PID : {pid}', f'Process Name', f'CPU Usage (%)', f'Memory Usage (%) ']
        writer.writerow(headers)

        # Write data by timestamp, ensuring each PID is only written once per timestamp
        for timestamp in sorted(processed_data.keys()):
            row = [timestamp]
            for pid in sorted_pids:
                if pid in processed_data[timestamp]:
                    process_name, cpu_usage, memory_usage = processed_data[timestamp][pid]
                    row += [pid, process_name, cpu_usage, memory_usage]
                else:
                    row += ['', '', '', '']  # Fill with empty values if no data for this PID
            writer.writerow(row)



import matplotlib.pyplot as plt
from collections import defaultdict
import os

def save_selected_process_cpu_mem(cpu_data, mem_data, time_data, save_path):
    """
    Updates data, calculates total CPU and memory usage for selected processes,
    plots these totals, and saves the plot to disk with filenames that include
    the process name and PID.

    Args:
    - cpu_data (dict): Dictionary storing lists of CPU usage per PID.
    - mem_data (dict): Dictionary storing lists of memory usage per PID.
    - time_data (dict): Dictionary storing lists of timestamp data per PID.
    - save_path (str): Directory to save the plots.
    """
    update_data_threads(process_names, cpu_data, mem_data, time_data)  # Updates the data collections
    plt.clf()  # Clear the current figure before plotting new data

    selected_options = print_selected_options()
    print("Selected options:", selected_options)
    if not selected_options:
        print("Selected options are not coming")
        return

    # Ensure the save path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Prepare data for total CPU and memory usage
    total_cpu_usage = {}
    total_mem_usage = {}
    
    for pid, cpus in cpu_data.items():
        process_name = get_process_name(pid)
        if process_name in selected_options:
            if process_name not in total_cpu_usage:
                total_cpu_usage[process_name] = 0
                total_mem_usage[process_name] = 0
            total_cpu_usage[process_name] += cpus[-1] if cpus else 0
            total_mem_usage[process_name] += mem_data[pid][-1] if mem_data[pid] else 0

            # Generate plot for each selected process
            plt.figure(figsize=(10, 5))  # New figure for each process

            # Plot CPU usage
            plt.subplot(2, 1, 1)
            plt.bar(process_name, total_cpu_usage[process_name], color='b')
            plt.title(f"CPU Usage for {process_name} (PID: {pid})")
            plt.ylabel("CPU Usage (%)")

            # Plot Memory usage
            plt.subplot(2, 1, 2)
            plt.bar(process_name, total_mem_usage[process_name], color='g')
            plt.title(f"Memory Usage for {process_name} (PID: {pid})")
            plt.ylabel("Memory Usage (%)")

            plt.tight_layout()  # Adjust layout to make sure everything fits without overlap

            # Save the figure to the specified path
            plot_filename = os.path.join(save_path, f"{process_name}_pid_{pid}.png")
            plt.savefig(plot_filename)
            print(f"Plot saved to {plot_filename}")

            plt.close()  # Close the plot to free up memory



# Function to update the plot with new data
def animate(i, fig, cpu_data, mem_data, time_data):
    update_data_threads(process_names, cpu_data, mem_data, time_data)
    plt.clf() # Clear the current figure
    selected_options = print_selected_options()
    print("Selected options:", selected_options) 
    if not selected_options:
        print("selected options are not coming")
        return
    
    for idx, (pid, data) in enumerate(cpu_data.items()):
        process_name = get_process_name(pid)
        if process_name in selected_options:
            plt.subplot(total_pids, 2, idx + 1)
            plt.subplots_adjust(hspace=0.9)
            plt.plot(time_data[pid], data)
            plt.title(f'CPU Usage for {process_name} (PID: {pid})')
            plt.xlabel('Time')
            plt.xticks(rotation=100)
            plt.ylabel('CPU Usage (%)')
            # plt.ylim(0, 100) 
        
    for idx, (pid, data) in enumerate(mem_data.items()):
        process_name = get_process_name(pid)
        if process_name in selected_options:
            plt.subplot(total_pids, 2, idx + 1 + total_pids)
            plt.plot(time_data[pid], data)
            plt.title(f'Memory Usage for {process_name} (PID: {pid})')
            plt.xlabel('Time')
            plt.ylabel('Memory Usage (%)')
            # plt.ylim(0, 100) 


    with open('process_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['PID', 'Process Name', 'Time', 'CPU Usage (%)', 'Memory Usage (%)'])
        for pid in cpu_data:
            process_name = get_process_name(pid)
            if process_name in selected_options:
                for i in range(len(cpu_data[pid])):
                    timestamp = datetime.fromtimestamp(time_data[pid][i]).strftime('%Y-%m-%d %H:%M:%S')
                    writer.writerow([pid, process_name, timestamp, cpu_data[pid][i], mem_data[pid][i]])
    
        

# Main loop
running = True
fig = plt.figure(figsize=(4, 5))
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Check if the event is handled by any checkbox
        event_handled = False
        for checkbox in checkboxes + [select_all_checkbox]:
            if checkbox.handle_event(event):
                event_handled = True
                break

        if not event_handled:
            # Handle other events here if needed
            pass

    screen.fill(WHITE)

    # Drawing checkboxes
    for checkbox in checkboxes + [select_all_checkbox]:
        checkbox.draw(screen, font)

    # Update display
    pygame.display.flip()
    clock.tick(60)

    # Update the plot
    animate(None, fig, cpu_data, mem_data, time_data)
    plt.draw()
    plt.pause(0.001)

# Quit Pygame

input_csv_path = 'process_data.csv'
output_csv_path = 'reformatted_process_data.csv'
save_path = './'
process_and_reformat_csv(input_csv_path, output_csv_path)
save_selected_process_cpu_mem(cpu_data, mem_data, time_data, save_path)

pygame.quit()
sys.exit()



import csv
import matplotlib.pyplot as plt

def read_process_data(filename):
    process_data = []
    with open(filename, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            process_data.append({
                'PID': int(row['PID']),
                'Process Name': row['Process Name'],
                'Time': row['Time'],
                'CPU Usage (%)': float(row['CPU Usage (%)']),
                'Memory Usage (%)': float(row['Memory Usage (%)'])
            })
    return process_data

def plot_process_data(process_data):
    unique_process_names = set([entry['Process Name'] for entry in process_data])
    for process_name in unique_process_names:
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        fig.suptitle(f'CPU and Memory Usage for {process_name}')
        for entry in process_data:
            if entry['Process Name'] == process_name:
                axs[0].plot(entry['Time'], entry['CPU Usage (%)'], label=f'PID: {entry["PID"]}')
                axs[1].plot(entry['Time'], entry['Memory Usage (%)'], label=f'PID: {entry["PID"]}')
        axs[0].set_ylabel('CPU Usage (%)')
        axs[1].set_ylabel('Memory Usage (%)')
        for ax in axs:
            ax.set_xlabel('Time')
            ax.legend()
            ax.grid(True)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

# Example usage:
filename = 'process_data.csv'
data = read_process_data(filename)
plot_process_data(data)



