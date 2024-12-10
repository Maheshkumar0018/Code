import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore')


def lat_lon_range(df_file):
	min_lat, max_lat = df_file['LAT'].min(),df_file['LAT'].max()
	min_lon, max_lon = df_file['LON'].min(), df_file['LON'].max()
	min_lat, max_lat, min_lon, max_lon = int(np.floor(min_lat)), int(np.ceil(max_lat)), int(np.floor(min_lon)), int(np.ceil(max_lon))
	return min_lat, max_lat, min_lon, max_lon

def compute_pattern_descriptor(segment):
    """
    Compute pattern descriptor for a given segment of trajectory.

    Parameters:
    - segment: DataFrame containing trajectory data.

    Returns:
    - descriptor: A string describing the movement pattern (example implementation).
    """
    avg_speed = segment['SOG'].mean()
    if avg_speed < 2:
        return "Stationary"
    elif avg_speed < 10:
        return "Slow Movement"
    else:
        return "High Speed"

def sliding_window_segmentation(data, window_size, step_size):
    """
    Perform sliding window segmentation on the given DataFrame.

    Parameters:
    - data: Pandas DataFrame containing the trajectory.
    - window_size: The number of data points in each sliding window.
    - step_size: The step size for the sliding window.

    Returns:
    - segments: A list of DataFrames, each representing a segmented window.
    """
    segments = []
    for start in range(0, len(data) - window_size + 1, step_size):
        window = data.iloc[start:start + window_size]
        segments.append(window)
    return segments


def prepare_ais_data_with_sliding_window_with_onehot_en(
    data, bounding_box, window_size, step_size, 
    resample_interval=10
):
    """
    Prepare AIS data using sliding window segmentation.

    Parameters:
    - data: Pandas DataFrame containing AIS data.
    - bounding_box: Tuple defining the bounding box (min_lat, max_lat, min_lon, max_lon).
    - window_size: The number of data points in each sliding window.
    - step_size: The step size for the sliding window.
    - resample_interval: Resampling interval in seconds.

    Returns:
    - combined_df: A single DataFrame containing all segmented trajectories and their features.
    """
    min_lat, max_lat, min_lon, max_lon = bounding_box

    # Step 1: Apply bounding box filter
    filtered_data = data[
        (data['LAT'] >= min_lat) & (data['LAT'] <= max_lat) &
        (data['LON'] >= min_lon) & (data['LON'] <= max_lon)
    ]

    # Step 2: Aggregate by MMSI and sort by timestamp
    grouped = filtered_data.groupby('MMSI')
    segments = []
    segment_id = 0

    # Step 3: Process each MMSI group
    for mmsi, group in tqdm(grouped, desc="Processing MMSI", unit="MMSI"):
        group = group.sort_values(by='BaseDateTime')
        group['BaseDateTime'] = pd.to_datetime(group['BaseDateTime'])

        # Step 4: Resample data
        timestamps = group['BaseDateTime']
        latitudes = group['LAT']
        longitudes = group['LON']
        sog = group['SOG']

        # If there is only one data point, skip interpolation
        if len(timestamps) < 2:
            print(f"Skipping MMSI {mmsi} due to insufficient data points")
            continue

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

        # Step 5: Perform sliding window segmentation
        trajectory_segments = sliding_window_segmentation(resampled_traj, window_size, step_size)

        for segment in trajectory_segments:
            # Compute pattern descriptor for each segment
            pattern_descriptor = compute_pattern_descriptor(segment)
            segment['PatternDescriptor'] = pattern_descriptor
            segment['MMSI'] = mmsi
            segment['SegmentID'] = segment_id
            
            # One-hot encode the pattern descriptor
            one_hot = pd.get_dummies(segment['PatternDescriptor'], prefix='Pattern')
            segment = pd.concat([segment, one_hot], axis=1)
            
            segments.append(segment)
            segment_id += 1

    # Combine all segmented trajectories into a single DataFrame
    combined_df = pd.concat(segments, ignore_index=True)
    return combined_df



def prepare_ais_data_with_sliding_window(
    data, bounding_box, window_size, step_size, 
    resample_interval=10
):
    """
    Prepare AIS data using sliding window segmentation.

    Parameters:
    - data: Pandas DataFrame containing AIS data.
    - bounding_box: Tuple defining the bounding box (min_lat, max_lat, min_lon, max_lon).
    - window_size: The number of data points in each sliding window.
    - step_size: The step size for the sliding window.
    - resample_interval: Resampling interval in seconds.

    Returns:
    - combined_df: A single DataFrame containing all segmented trajectories and their features.
    """
    min_lat, max_lat, min_lon, max_lon = bounding_box

    # Step 1: Apply bounding box filter
    filtered_data = data[
        (data['LAT'] >= min_lat) & (data['LAT'] <= max_lat) &
        (data['LON'] >= min_lon) & (data['LON'] <= max_lon)
    ]

    # Step 2: Aggregate by MMSI and sort by timestamp
    grouped = filtered_data.groupby('MMSI')
    segments = []
    segment_id = 0

    # Step 3: Process each MMSI group
    for mmsi, group in tqdm(grouped, desc="Processing MMSI", unit="MMSI"):
        group = group.sort_values(by='BaseDateTime')
        group['BaseDateTime'] = pd.to_datetime(group['BaseDateTime'])

        # Step 4: Resample data
        timestamps = group['BaseDateTime']
        latitudes = group['LAT']
        longitudes = group['LON']
        sog = group['SOG']

        # If there is only one data point, skip interpolation
        if len(timestamps) < 2:
            print(f"Skipping MMSI {mmsi} due to insufficient data points")
            continue

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

        # Step 5: Perform sliding window segmentation
        trajectory_segments = sliding_window_segmentation(resampled_traj, window_size, step_size)

        for segment in trajectory_segments:
            # Compute pattern descriptor for each segment
            pattern_descriptor = compute_pattern_descriptor(segment)
            segment['PatternDescriptor'] = pattern_descriptor
            segment['MMSI'] = mmsi
            segment['SegmentID'] = segment_id
            segments.append(segment)
            segment_id += 1

    # Combine all segmented trajectories into a single DataFrame
    combined_df = pd.concat(segments, ignore_index=True)
    return combined_df


# Example usage:
if __name__ == "__main__":

    file_path = '/home/mglocadmin/Mahesh/AIS_2022_03_31.csv'
    ais_data = pd.read_csv(file_path)  
    min_lat, max_lat, min_lon, max_lon = lat_lon_range(ais_data)
    bounding_box = (min_lat, max_lat, min_lon, max_lon)  

    window_size = 10  
    step_size = 5    

    # Process AIS data with sliding window segmentation
    segmented_trajectories_df = prepare_ais_data_with_sliding_window(
        ais_data, bounding_box, window_size, step_size
    )

    # Save segmented DataFrame to a CSV
    segmented_trajectories_df.to_csv('./segmented_trajectories.csv', index=False)
    print("Segmentation completed and saved to 'segmented_trajectories.csv'")
