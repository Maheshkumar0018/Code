import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.interpolate import interp1d
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compute_pattern_descriptor(trajectory):
    """
    Compute a pattern descriptor for a given trajectory.
    Example: classify patterns based on average speed.

    Parameters:
    - trajectory: Pandas DataFrame containing a single trajectory.

    Returns:
    - descriptor: String or numerical value representing the pattern.
    """
    avg_speed = trajectory['SOG'].mean()
    if avg_speed < 1:
        return "Stationary"
    elif avg_speed < 5:
        return "Slow Movement"
    else:
        return "High Speed"

def prepare_ais_data(
    data, bounding_box, specific_mmsi=None, time_gap_threshold=timedelta(minutes=10), 
    resample_interval=10
):
    """
    Prepare AIS data with filtering, trajectory processing, interpolation, and windowing.

    Parameters:
    - data: Pandas DataFrame containing AIS data.
    - bounding_box: Tuple defining the bounding box (min_lat, max_lat, min_lon, max_lon).
    - specific_mmsi: Specific MMSI to filter data for (optional).
    - time_gap_threshold: Threshold for splitting sub-trajectories (timedelta).
    - resample_interval: Resampling interval in seconds.

    Returns:
    - combined_df: A single DataFrame containing all processed trajectories and their features.
    """
    min_lat, max_lat, min_lon, max_lon = bounding_box
    
    # Step 1: Apply bounding box filter
    filtered_data = data[
        (data['LAT'] >= min_lat) & (data['LAT'] <= max_lat) &
        (data['LON'] >= min_lon) & (data['LON'] <= max_lon)
    ]
    
    # Step 2: Filter for specific MMSI if provided
    if specific_mmsi:
        filtered_data = filtered_data[filtered_data['MMSI'] == specific_mmsi]
    
    # Step 3: Aggregate by MMSI and sort by timestamp
    grouped = filtered_data.groupby('MMSI')
    trajectories = []
    trajectory_id = 0  # Unique ID for each trajectory
    
    # Step 4: Add progress bar for MMSI processing
    for mmsi, group in tqdm(grouped, desc="Processing MMSI", unit="MMSI"):
        group = group.sort_values(by='BaseDateTime')
        group['BaseDateTime'] = pd.to_datetime(group['BaseDateTime'])
        
        # Step 5: Split into sub-trajectories based on time gap threshold
        group['time_diff'] = group['BaseDateTime'].diff()
        sub_trajectories = np.split(group, group[group['time_diff'] > time_gap_threshold].index)
        
        # Add progress bar for sub-trajectories
        for traj in tqdm(sub_trajectories, desc=f"Processing Sub-trajectories for MMSI {mmsi}", leave=False):
            if len(traj) > 1:
                # Step 6: Linear interpolation
                traj = traj.reset_index(drop=True)
                timestamps = traj['BaseDateTime']
                latitudes = traj['LAT']
                longitudes = traj['LON']
                sog = traj['SOG']

                # Generate new timestamps for resampling
                start_time = timestamps.iloc[0]
                end_time = timestamps.iloc[-1]
                new_timestamps = pd.date_range(start=start_time, end=end_time, freq=f'{resample_interval}S')

                # Interpolation
                interp_lat = interp1d(
                    timestamps.astype(np.int64), latitudes, kind='linear', bounds_error=False, fill_value="extrapolate"
                )
                interp_lon = interp1d(
                    timestamps.astype(np.int64), longitudes, kind='linear', bounds_error=False, fill_value="extrapolate"
                )
                interp_sog = interp1d(
                    timestamps.astype(np.int64), sog, kind='linear', bounds_error=False, fill_value="extrapolate"
                )
                
                resampled_traj = pd.DataFrame({
                    'BaseDateTime': new_timestamps,
                    'LAT': interp_lat(new_timestamps.astype(np.int64)),
                    'LON': interp_lon(new_timestamps.astype(np.int64)),
                    'SOG': interp_sog(new_timestamps.astype(np.int64))
                })

                # Step 7: Annotate with pattern descriptor
                pattern_descriptor = compute_pattern_descriptor(resampled_traj)
                resampled_traj['PatternDescriptor'] = pattern_descriptor
                resampled_traj['MMSI'] = mmsi
                resampled_traj['TrajectoryID'] = trajectory_id

                # Add the processed trajectory
                trajectories.append(resampled_traj)
                trajectory_id += 1

    # Combine all trajectories into a single DataFrame
    combined_df = pd.concat(trajectories, ignore_index=True)
    return combined_df

# ais_data = pd.read_csv('/home/mglocadmin/Mahesh/AIS_2022_03_31.csv')  
# bounding_box = (25.0, 50.0, -130.0, -60.0)  
# all_trajectories_df = prepare_ais_data(ais_data, bounding_box)
# all_trajectories_df.to_csv('./all_trajectories.csv', index=False)



def save_trajectories_plot(data, filename='trajectories_plot.png'):
    """
    Plot all trajectories in the dataset and save the plot as an image.
    
    Parameters:
    - data: Pandas DataFrame containing AIS data with columns 'LAT', 'LON', and 'TrajectoryID'.
    - filename: String representing the file path and name to save the plot (default: 'trajectories_plot.png').
    
    Returns:
    - None: Saves the plot as an image file.
    """
    # Create a unique set of trajectory IDs
    unique_trajectory_ids = data['TrajectoryID'].unique()
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Loop over each trajectory and plot its path
    for trajectory_id in unique_trajectory_ids:
        # Filter data for the current trajectory
        trajectory_data = data[data['TrajectoryID'] == trajectory_id]
        
        # Assign a unique color for each trajectory
        color = plt.cm.jet(np.random.rand())  # Use random colors
        
        # Plot the trajectory with the selected color
        plt.plot(trajectory_data['LON'], trajectory_data['LAT'], label=f"Trajectory {trajectory_id}", color=color)
    
    # Customize plot appearance
    plt.title("Trajectories of Vessels")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(title="Trajectory ID")
    plt.grid(True)
    
    # Save the plot to a file
    plt.savefig(filename)
    
    # Optionally, display the plot (if needed)
    # plt.show()



def save_individual_trajectories(data, output_folder='./'):
    """
    Save each individual trajectory as a separate plot.
    
    Parameters:
    - data: Pandas DataFrame containing AIS data with columns 'LAT', 'LON', and 'TrajectoryID'.
    - output_folder: Folder where individual trajectory plots will be saved (default: './').
    
    Returns:
    - None: Saves each trajectory as an individual image file.
    """
    # Create a unique set of trajectory IDs
    unique_trajectory_ids = data['TrajectoryID'].unique()
    # if trajectory_ids is None:
    #     trajectory_ids = unique_trajectory_ids[:2]  
    
    # Loop over each trajectory and save its plot
    for trajectory_id in unique_trajectory_ids:
        # Filter data for the current trajectory
        trajectory_data = data[data['TrajectoryID'] == trajectory_id]
        
        # Set up the plot
        plt.figure(figsize=(12, 8))
        
        # Plot the trajectory
        plt.plot(trajectory_data['LON'], trajectory_data['LAT'], label=f"Trajectory {trajectory_id}")
        
        # Customize plot appearance
        plt.title(f"Trajectory {trajectory_id}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend(title="Trajectory ID")
        plt.grid(True)
        
        # Define the filename for saving the plot
        output_filename = f"{output_folder}/trajectory_{trajectory_id}.png"
        
        # Save the plot as an image file
        plt.savefig(output_filename)
        plt.close()  # Close the figure to prevent memory issues


ais_data = pd.read_csv('./all_trajectories.csv')  

save_individual_trajectories(ais_data, output_folder='./trajectories')

