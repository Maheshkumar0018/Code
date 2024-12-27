import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# Add temporal features
def add_temporal_features(data, time_col):
    data[time_col] = pd.to_datetime(data[time_col])
    data['Hour'] = data[time_col].dt.hour
    data['Minute'] = data[time_col].dt.minute
    data['Second'] = data[time_col].dt.second
    data['TimeDelta'] = data[time_col].diff().dt.total_seconds().fillna(0)
    return data


# Filter segments with insufficient rows
def filter_short_segments(data, window_size, prediction_horizon, group_col='SegmentID'):
    min_required_rows = window_size + prediction_horizon
    filtered_data = data.groupby(group_col).filter(lambda x: len(x) >= min_required_rows)
    print(f"Filtered data contains {len(filtered_data)} rows after removing short segments.")
    return filtered_data


# Prepare sequences with temporal features and SegmentID
def prepare_sequences(data, window_size, prediction_horizon,
                      group_col='SegmentID', time_col='BaseDateTime'):
    sequences = []
    unique_segments = sorted(data[group_col].unique())

    for segment in tqdm(unique_segments, desc="Processing Segments", unit="segment"):
        # print('************',segment)
        segment_data = data[data[group_col] == segment].reset_index(drop=True)
        num_rows = len(segment_data)
        # print(f"Segment {segment} has {num_rows} rows.")

        if num_rows < window_size + prediction_horizon:
            # print(f"Skipping segment {segment}: Not enough rows.")
            continue

        for start in range(num_rows - window_size - prediction_horizon + 1):
            x_window = segment_data.iloc[start:start+window_size]
            y_window = segment_data.iloc[start+window_size:start+window_size+prediction_horizon]
            sequences.append({
                'X': {
                    'LAT': x_window['LAT'].tolist(),
                    'LON': x_window['LON'].tolist(),
                    'SOG': x_window['SOG'].tolist(),
                    'Hour': x_window['Hour'].tolist(),
                    'Minute': x_window['Minute'].tolist(),
                    'Second': x_window['Second'].tolist(),
                    'TimeDelta': x_window['TimeDelta'].tolist(),
                    'SegmentID': x_window['SegmentID'].tolist(),
                },
                'Y': {
                    'LAT': y_window['LAT'].tolist(),
                    'LON': y_window['LON'].tolist(),
                    'SOG': y_window['SOG'].tolist(),
                    'Hour': y_window['Hour'].tolist(),
                    'Minute': y_window['Minute'].tolist(),
                    'Second': y_window['Second'].tolist(),
                    'TimeDelta': y_window['TimeDelta'].tolist(),
                    'SegmentID': y_window['SegmentID'].tolist(),
                }
            })
    return sequences


# Split data into train/val/test
# def split_train_val_test(data, test_size=0.2, val_size=0.2):
#     train_data, temp_data = train_test_split(data, test_size=test_size + val_size, random_state=42, shuffle=True)
#     val_data, test_data = train_test_split(temp_data, test_size=test_size / (test_size + val_size), random_state=42, shuffle=True)
#     return train_data, val_data, test_data

# def split_train_val_test(data, test_size=0.2, val_size=0.2, group_col='SegmentID'):
#     sorted_segments = sorted(data[group_col].unique())  # Sort SegmentIDs
#     train_segments, temp_segments = train_test_split(sorted_segments, test_size=test_size + val_size, random_state=42)
#     val_segments, test_segments = train_test_split(temp_segments, test_size=test_size / (test_size + val_size), random_state=42)

#     train_data = data[data[group_col].isin(train_segments)]
#     val_data = data[data[group_col].isin(val_segments)]
#     test_data = data[data[group_col].isin(test_segments)]

#     return train_data, val_data, test_data

def split_within_segments(data, train_size=0.7, val_size=0.1, test_size=0.2, group_col='SegmentID'):
    train_data = []
    val_data = []
    test_data = []
    
    unique_segments = sorted(data[group_col].unique())  # Sort SegmentIDs
    
    for segment in unique_segments:
        segment_data = data[data[group_col] == segment]
        
        # Split into train and temp (validation + test)
        train_segment, temp_segment = train_test_split(
            segment_data, test_size=(val_size + test_size), random_state=42, shuffle=False
        )
        
        # Further split temp_segment into validation and test
        val_segment, test_segment = train_test_split(
            temp_segment, test_size=(test_size / (val_size + test_size)), random_state=42, shuffle=False
        )
        
        # Append splits to their respective lists
        train_data.append(train_segment)
        val_data.append(val_segment)
        test_data.append(test_segment)
    
    # Concatenate all segments' splits
    train_data = pd.concat(train_data).reset_index(drop=True)
    val_data = pd.concat(val_data).reset_index(drop=True)
    test_data = pd.concat(test_data).reset_index(drop=True)
    
    return train_data, val_data, test_data

# Data preparation pipeline without scaling
def prepare_data_pipeline_no_scaling(data, window_size, prediction_horizon,
                                     test_size=0.2, val_size=0.2):
    # Add temporal features
    print("Adding temporal features...")
    data = add_temporal_features(data, 'BaseDateTime')

    # Filter short segments
    print("Filtering short segments...")
    data = filter_short_segments(data, window_size, prediction_horizon)

    # Split data
    print("Splitting data into train, validation, and test sets...")
    train_data, val_data, test_data = split_train_val_test(data, test_size, val_size)

    # Prepare sequences
    print("Preparing training sequences...")
    train_sequences = prepare_sequences(train_data, window_size, prediction_horizon)

    print("Preparing validation sequences...")
    val_sequences = prepare_sequences(val_data, window_size, prediction_horizon)

    print("Preparing test sequences...")
    test_sequences = prepare_sequences(test_data, window_size, prediction_horizon)

    return train_sequences, val_sequences, test_sequences

# Parameters
window_size = 5
prediction_horizon = 2
test_size = 0.2
val_size = 0.2


# Prepare data
train_sequences, val_sequences, test_sequences = prepare_data_pipeline_no_scaling(
    segmented_trajectories_df, window_size, prediction_horizon, test_size, val_size
)

print(f"Number of training sequences: {len(train_sequences)}")
print(f"Number of validation sequences: {len(val_sequences)}")
print(f"Number of test sequences: {len(test_sequences)}")

