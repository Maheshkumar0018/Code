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
import matplotlib.dates as mdates
import os

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
WHITE = (255, 255, 255)
FONT_SIZE = 24

DATA_UPDATE_INTERVAL = 10  # Set the delay interval in seconds
CSV_SAVE_INTERVAL = 1  # Set the interval for saving data to CSV in seconds

# Set up the Pygame window
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Process Monitor")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, FONT_SIZE)

# Locks for thread safety
cpu_data_lock = threading.Lock()
mem_data_lock = threading.Lock()
time_data_lock = threading.Lock()

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
    while True:
        current_time = time.time()  # Fetch current time outside the loop
        start_time = current_time - 120  # 300 seconds for 5 minutes

        for process in psutil.process_iter():
            try:
                name = process.name()
                pid = process.pid
                if process_name.lower() in name.lower():
                    cpu_percent = process.cpu_percent(interval=None)
                    mem_percent = process.memory_percent()
                    # current_time = time.time()

                    with cpu_data_lock:
                        cpu_data[pid].append(cpu_percent)
                    with mem_data_lock:
                        mem_data[pid].append(mem_percent)
                    with time_data_lock:
                        time_data[pid].append(current_time)  # Store the timestamp

                    # Trim the data to the latest 5 minutes
                    while time_data[pid] and time_data[pid][0] < start_time:
                        cpu_data[pid].pop(0)
                        mem_data[pid].pop(0)
                        time_data[pid].pop(0)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

        time.sleep(DATA_UPDATE_INTERVAL)  # Add delay here

# Function to update CPU and memory usage data using threads
def update_data_threads(process_names, cpu_data, mem_data, time_data):
    threads = []
    for process_name in process_names:
        thread = threading.Thread(target=update_data, args=(process_name, cpu_data, mem_data, time_data))
        threads.append(thread)
        thread.start()
    return threads


def save_data_to_csv(cpu_data, mem_data, time_data, total_cores, total_ram):
    try:
        with cpu_data_lock, mem_data_lock, time_data_lock:
            if not os.path.exists('process_data.csv'):
                # Create the file and write the header row
                with open('process_data.csv', mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['PID', 'Process Name', 'Time', 'CPU Usage (%)', 'Memory Usage (%)', 'Total Cores', 'Total RAM'])

            with open('process_data.csv', mode='a', newline='') as file:  # Open in append mode
                writer = csv.writer(file)
                for pid in cpu_data:
                    process_name = get_process_name(pid)
                    for i in range(len(cpu_data[pid])):
                        timestamp = datetime.fromtimestamp(time_data[pid][i]).strftime('%Y-%m-%d %H:%M:%S')
                        writer.writerow([pid, process_name, timestamp, cpu_data[pid][i], mem_data[pid][i], total_cores, total_ram])
    except Exception as e:
        print(f"Error saving data to CSV: {e}")


def periodic_csv_save():
    while True:
        total_cores, total_ram = get_system_info()
        save_data_to_csv(cpu_data, mem_data, time_data, total_cores, total_ram)
        time.sleep(CSV_SAVE_INTERVAL)

def get_system_info():
    total_cores = psutil.cpu_count(logical=False)  # Get total number of physical cores
    total_ram = psutil.virtual_memory().total  # Get total RAM in bytes
    formatted_cores = f"{total_cores} cores"
    formatted_ram = convert_bytes_to_gb(total_ram)
    return formatted_cores, formatted_ram

def convert_bytes_to_gb(bytes_value):
    gb_value = bytes_value / (1024 ** 3)  # Convert bytes to gigabytes
    formatted_gb = f"{gb_value:.2f} GB"
    return formatted_gb

def animate(i, fig, cpu_data, mem_data, time_data):
    try:
        plt.clf()  # Clear the current figure
        selected_options = print_selected_options()
        total_cores, total_ram = get_system_info()
        # print("Selected options:", selected_options)
        if not selected_options:
            # print("selected options are not coming")
            return

        total_pids = max(len(cpu_data), len(mem_data))

        fig.subplots_adjust(hspace=0.5)  # Adjust vertical spacing between subplots

        # Determine the time interval to display (e.g., past 1 minute)
        current_time = time.time()
        start_time = current_time - 60  # 60 seconds for 1 minute
        end_time = current_time

        for idx, (pid, data) in enumerate(cpu_data.items()):
            process_name = get_process_name(pid)
            if process_name in selected_options:
                ax = fig.add_subplot(total_pids, 2, idx * 2 + 1)
                formatted_time = [datetime.fromtimestamp(ts) for ts in time_data[pid]]

                # Filter data within the time interval
                filtered_data = []
                filtered_time = []
                for ts, value in zip(time_data[pid], data):
                    if start_time <= ts <= end_time:
                        filtered_data.append(value)
                        # filtered_time.append(datetime.fromtimestamp(ts))
                        filtered_time.append(ts - start_time)

                # Retain a certain amount of historical data within the time interval
                max_data_points = 200  # Adjust as needed
                if len(filtered_data) > max_data_points:
                    filtered_data = filtered_data[-max_data_points:]
                    filtered_time = filtered_time[-max_data_points:]

                ax.plot(filtered_time, filtered_data)
                ax.set_title(f'CPU Usage for {process_name} (PID: {pid})')
                ax.set_xlabel('Time')
                ax.tick_params(axis='x', labelsize=7)
                ax.set_ylabel('CPU Usage (%)')
                # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Hh:%Mm:%Ss'))
                plt.xticks(rotation=50)

        for idx, (pid, data) in enumerate(mem_data.items()):
            process_name = get_process_name(pid)
            if process_name in selected_options:
                ax = fig.add_subplot(total_pids, 2, idx * 2 + 2)
                formatted_time = [datetime.fromtimestamp(ts) for ts in time_data[pid]]

                # Filter data within the time interval
                filtered_data = []
                filtered_time = []
                for ts, value in zip(time_data[pid], data):
                    if start_time <= ts <= end_time:
                        filtered_data.append(value)
                        # filtered_time.append(datetime.fromtimestamp(ts))
                        filtered_time.append(ts - start_time)

                # Retain a certain amount of historical data within the time interval
                max_data_points = 100  # Adjust as needed
                if len(filtered_data) > max_data_points:
                    filtered_data = filtered_data[-max_data_points:]
                    filtered_time = filtered_time[-max_data_points:]

                ax.plot(filtered_time, filtered_data)
                ax.set_title(f'Memory Usage for {process_name} (PID: {pid})')
                ax.set_xlabel('Time')
                ax.tick_params(axis='x', labelsize=7)
                ax.set_ylabel('Memory Usage (%)')
                # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Hh:%Mm:%Ss'))
                plt.xticks(rotation=50)
    except Exception as e:
        print(f"Error in animate function: {e}")


# Main loop
def main_loop():
    running = True
    fig = plt.figure(figsize=(4, 5))
    while running:
        try:
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
        except Exception as e:
            print(f"Error in main loop: {e}")
            running = False

    pygame.quit()
    sys.exit()

# Start data collection threads
threads = update_data_threads(process_names, cpu_data, mem_data, time_data)

# Start periodic CSV save thread
csv_thread = threading.Thread(target=periodic_csv_save)
csv_thread.daemon = True
csv_thread.start()

# Run the main loop
main_loop()
