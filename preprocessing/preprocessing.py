import mne
import numpy as np
import pandas as pd
import re
import logging
import pickle

from matplotlib import pyplot as plt
import seaborn as sns

from brainvision import read_raw_brainvision as read_raw_brainvision_local

unified_events_dict = {
    'Time 0/': 0,
    'Stimulus/ 3-B-NOSTOP-L': 1,
    'Stimulus/ 3-B-NOSTOP-R': 2,

    'Stimulus/ 3-B-STOP1-SE-L': 3,
    'Stimulus/ 3-B-STOP1-SE-R': 4,
    'Stimulus/ 3-B-STOP1-SS-L': 5,
    'Stimulus/ 3-B-STOP1-SS-R': 6,

    'Stimulus/ 3-B-STOP2-SE-L': 7,
    'Stimulus/ 3-B-STOP2-SE-R': 8,
    'Stimulus/ 3-B-STOP2-SS-L': 9,
    'Stimulus/ 3-B-STOP2-SS-R': 10,

    'Stimulus/ 3-B-STOP3-SE-L': 11,
    'Stimulus/ 3-B-STOP3-SE-R': 12,
    'Stimulus/ 3-B-STOP3-SS-L': 13,
    'Stimulus/ 3-B-STOP3-SS-R': 14,

    'Stimulus/ 3-B-STOP4-SE-L': 15,
    'Stimulus/ 3-B-STOP4-SE-R': 16,
    'Stimulus/ 3-B-STOP4-SS-L': 17,
    'Stimulus/ 3-B-STOP4-SS-R': 18,

    'Stimulus/ 3-B-STOP5-SE-L': 19,
    'Stimulus/ 3-B-STOP5-SE-R': 20,
    'Stimulus/ 3-B-STOP5-SS-L': 21,
    'Stimulus/ 3-B-STOP5-SS-R': 22,

    'Stimulus/ 3-B-STOP6-SE-L': 23,
    'Stimulus/ 3-B-STOP6-SE-R': 24,
    'Stimulus/ 3-B-STOP6-SS-L': 25,
    'Stimulus/ 3-B-STOP6-SS-R': 26,

    'Stimulus/ 3-B-STOP7-SE-L': 27,
    'Stimulus/ 3-B-STOP7-SE-R': 28,
    'Stimulus/ 3-B-STOP7-SS-L': 29,
    'Stimulus/ 3-B-STOP7-SS-R': 30,

    'Stimulus/ 3-R-B-NOSTOP-L': 31,
    'Stimulus/ 3-R-B-NOSTOP-R': 32,

    'Stimulus/ 3-R-B-STOP1-SE-L': 33,
    'Stimulus/ 3-R-B-STOP1-SE-R': 34,

    'Stimulus/ 3-R-B-STOP2-SE-L': 35,
    'Stimulus/ 3-R-B-STOP2-SE-R': 36,

    'Stimulus/ 3-R-B-STOP3-SE-L': 37,
    'Stimulus/ 3-R-B-STOP3-SE-R': 38,

    'Stimulus/ 3-R-B-STOP4-SE-L': 39,
    'Stimulus/ 3-R-B-STOP4-SE-R': 40,

    'Stimulus/ 3-R-B-STOP5-SE-L': 41,
    'Stimulus/ 3-R-B-STOP5-SE-R': 42,

    'Stimulus/ 3-R-B-STOP6-SE-L': 43,
    'Stimulus/ 3-R-B-STOP6-SE-R': 44,

    'Stimulus/ 3-R-B-STOP7-SE-L': 45,
    'Stimulus/ 3-R-B-STOP7-SE-R': 46,

    'Stimulus/ 3-STOP1-SE-L': 47,
    'Stimulus/ 3-STOP1-SE-R': 48,
    'Stimulus/ 3-STOP1-SS-L': 49,
    'Stimulus/ 3-STOP1-SS-R': 50,

    'Stimulus/ 3-STOP2-SE-L': 51,
    'Stimulus/ 3-STOP2-SE-R': 52,
    'Stimulus/ 3-STOP2-SS-L': 53,
    'Stimulus/ 3-STOP2-SS-R': 54,

    'Stimulus/ 3-STOP3-SE-L': 55,
    'Stimulus/ 3-STOP3-SE-R': 56,
    'Stimulus/ 3-STOP3-SS-L': 57,
    'Stimulus/ 3-STOP3-SS-R': 58,

    'Stimulus/ 3-STOP4-SE-L': 59,
    'Stimulus/ 3-STOP4-SE-R': 60,
    'Stimulus/ 3-STOP4-SS-L': 61,
    'Stimulus/ 3-STOP4-SS-R': 62,

    'Stimulus/ 3-STOP5-SE-L': 63,
    'Stimulus/ 3-STOP5-SE-R': 64,
    'Stimulus/ 3-STOP5-SS-L': 65,
    'Stimulus/ 3-STOP5-SS-R': 66,

    'Stimulus/ 3-STOP6-SE-L': 67,
    'Stimulus/ 3-STOP6-SE-R': 68,
    'Stimulus/ 3-STOP6-SS-L': 69,
    'Stimulus/ 3-STOP6-SS-R': 70,

    'Stimulus/ 3-STOP7-SE-L': 71,
    'Stimulus/ 3-STOP7-SE-R': 72,
    'Stimulus/ 3-STOP7-SS-L': 73,
    'Stimulus/ 3-STOP7-SS-R': 74,
}


def load_raw_brainvision(
        file_name,
        dir_name,
        unified_events_dict,
        logger_preprocessing_info,
        picks=None,
        tmin=-0.2,
        tmax=1.58,
):
    logger_preprocessing_info.info(f'###### Reading file: {file_name}')

    raw = mne.io.read_raw_brainvision(f'{dir_name}/{file_name}.vhdr', preload=True, eog=('HEOG', 'VEOG'))
    sampling_rate = raw.info['sfreq']
    logger_preprocessing_info.info(f'Raw shape: {raw.get_data().shape}')
    events, event_id = mne.events_from_annotations(raw, unified_events_dict)

    epochs = mne.Epochs(
        raw,
        tmin=tmin,
        tmax=tmax,
        events=events,
        event_id=[0],
        baseline=None,
        preload=True,
        picks=picks,
    )

    logger_preprocessing_info.info(f'Epochs shape: {epochs.get_data(copy=True).shape}')

    return epochs, events, event_id, sampling_rate


def artifact_rejection(
        epochs,
        sampling_rate,
        logger_preprocessing_info,
        vmin=-100e-6,
        vmax=100e-6,
        baseline_indices=(-0.2, 0)
):
    data_epochs = epochs.get_data(copy=True)
    channels = epochs.info['ch_names']
    baseline_channels_dict = {}
    drop_log = {}
    drop_log_full = {}

    # Iterate over trials and channels
    for trial_index, trial in enumerate(data_epochs):
        for channel_index, trial_channel in enumerate(trial):
            # Calculate baseline range
            start = int(abs((epochs.tmin - baseline_indices[0])) * sampling_rate)
            length = baseline_indices[1] - baseline_indices[0]
            stop = int(length * sampling_rate)

            # Calculate baseline for this trial-channel
            baseline = trial_channel[start:stop].mean()

            # Store baseline
            if channels[channel_index] not in baseline_channels_dict:
                baseline_channels_dict[channels[channel_index]] = {}
            baseline_channels_dict[channels[channel_index]][trial_index] = baseline

            # Apply baseline correction
            trial_channel -= baseline

            # Check for amplitude exceeding +-100 µV (0.0001 V)
            if (trial_channel > vmax).any() or (trial_channel < vmin).any():
                logger_preprocessing_info.info(
                    f'###### DROP LOG: Trial {trial_index} at {channels[channel_index]} channel to drop')
                if channels[channel_index] not in drop_log:
                    drop_log[channels[channel_index]] = []
                drop_log[channels[channel_index]].append(trial_index)

            # Store drop log
            if (trial_channel > vmax).any() or (trial_channel < vmin).any():
                if channels[channel_index] not in drop_log_full:
                    drop_log_full[channels[channel_index]] = {}
                drop_log_full[channels[channel_index]][trial_index] = True
            else:
                if channels[channel_index] not in drop_log_full:
                    drop_log_full[channels[channel_index]] = {}
                drop_log_full[channels[channel_index]][trial_index] = False

    return baseline_channels_dict, drop_log, drop_log_full


def map_event(event):
    # Patterns for matching
    patterns = {
        r'^Stimulus/ 3-B-NOSTOP.*': 'go/nostop',
        r'^Stimulus/ 3-B-STOP(\d+)-(SS|SE).*': lambda m: f'go/stop/{m.group(1)}/{m.group(2)}',
        r'^Stimulus/ 3-R-B-NOSTOP.*': 'response/correct',
        r'^Stimulus/ 3-R-B-STOP(\d+).*': lambda m: f'response/incorrect/{m.group(1)}',
        r'^Stimulus/ 3-STOP(\d+)-(SS|SE).*': lambda m: f'stop/{m.group(1)}/{m.group(2)}'
    }
    # Check each pattern
    for pattern, replacement in patterns.items():
        match = re.fullmatch(pattern, event)
        if match:
            return replacement if not callable(replacement) else replacement(match)
    # Default return value if no pattern matches
    return 'unknown'


# Define the function to categorize events into 'go', 'response', or 'stop'
def categorize_type(event_general):
    if 'go' in event_general:
        return 'go'
    elif 'response/correct' in event_general:
        return 'response_nostop'
    elif 'response' in event_general:
        return 'response_stop'
    elif 'stop' in event_general:
        return 'stop'
    return 'unknown'


# Define the SSD mapping
ssd_mapping = {
    1: 100,
    2: 150,
    3: 200,
    4: 250,
    5: 300,
    6: 350,
    7: 400
}


# Define the function to map to SSD values
def map_ssd(event_general, fill_missing=True):
    """
    Map SSD based on the event string.

    Parameters:
    event_general (str): The event string.
    fill_missing (bool): If True, fill missing values with 0, otherwise with np.nan.

    Returns:
    float: The mapped SSD value or np.nan if not found.
    """
    # Define the patterns to match
    patterns = [
        r'go/stop/(\d+)',
        r'stop/(\d+)',
        r'response/incorrect/(\d+)'
    ]

    for pattern in patterns:
        match = re.search(pattern, event_general)
        if match:
            number = int(match.group(1))
            return ssd_mapping.get(number, 0 if fill_missing else np.nan)

    # If no match is found, return the appropriate missing value
    return 0 if fill_missing else np.nan


def add_sri(
        df,
        logger_preprocessing_info,
        sampling_rate,
        fill_missing=True
):
    """
    Add the SRI column to the DataFrame, which represents the distance
    between stop events and response/incorrect events in terms of latency
    multiplied by the sampling rate.

    Parameters:
    df (pd.DataFrame): DataFrame containing events with 'event' and 'latency' columns.
    sampling_rate (float): The sampling rate to convert latency differences to time.

    Returns:
    pd.DataFrame: DataFrame with an additional 'sri' column.
    """

    # Initialize the sri column with np.nan
    df['sri'] = 0.0 if fill_missing else np.nan
    df = df.reset_index()

    # Iterate through the DataFrame to calculate the sri values
    for idx, row in df.iterrows():
        if 'response/incorrect' in row['event_general']:
            stop_idx = idx - 1
            # Adjust stop_idx if the previous event is 'unknown'
            if stop_idx > 0 and 'unknown' in df.loc[stop_idx, 'event_general']:
                logger_preprocessing_info.info(f'Time 0 at STOP index {stop_idx}. Checking index-2 {idx - 2}')
                stop_idx -= 1
            # Check if stop_idx is valid and the event is 'stop'
            if stop_idx > 0:
                if 'stop' in df.loc[stop_idx, 'type']:
                    logger_preprocessing_info.debug(f'passed even_type is stop at {stop_idx}')
                    distance = (row['latency'] - df.loc[stop_idx, 'latency']) / sampling_rate * 1000
                    df.at[idx, 'sri'] = distance
                else:
                    logger_preprocessing_info.info(f'No stop above error/incorrect at index {idx}')
            else:
                logger_preprocessing_info.info('STOP idx at <= 0')

    return df


def create_events_dataframe(
        events,
        event_id,
        sampling_rate,
        logger_preprocessing_info,
        fill_missing=False
):
    events_df = pd.DataFrame(events, columns=['latency', 'duration', 'id'])

    # Invert the dictionary to map IDs to event names
    id_to_event = {value: key for key, value in event_id.items()}

    # Create a new 'event' column by mapping 'id' to event names
    events_df['event'] = events_df['id'].map(id_to_event)
    events_df['event_general'] = events_df['event'].apply(map_event)
    events_df['type'] = events_df['event_general'].apply(categorize_type)

    # add info on SSD
    events_df['ssd'] = events_df['event_general'].apply(map_ssd, fill_missing=fill_missing)

    # add info on response type
    events_df['response_type'] = events_df['event_general'].str.extract(r'response/(correct|incorrect)', expand=False)
    events_df['response_type'] = events_df['response_type'].fillna('n-a')

    # add info on stop type
    events_df['stop_type'] = events_df['event_general'].str.extract(r'stop/.*/(SE|SS)', expand=False)
    events_df['stop_type'] = events_df['stop_type'].fillna('n-a')

    # add info on SRI
    events_df = add_sri(
        events_df,
        logger_preprocessing_info=logger_preprocessing_info,
        sampling_rate=sampling_rate,
        fill_missing=fill_missing
    )

    # # center and normalize continuous predictors
    # ssd_mean = np.nanmean(events_df['ssd'])
    # events_df['ssd_centered'] = events_df['ssd'] - ssd_mean
    # events_df['ssd_normalized'] = events_df['ssd_centered'] / np.nanstd(events_df['ssd_centered'])
    # events_df['ssd_standardized'] = events_df['ssd'] / np.nanstd(events_df['ssd'])
    #
    # sri_mean = np.nanmean(events_df['sri'])
    # events_df['sri_centered'] = events_df['sri'] - sri_mean
    # events_df['sri_normalized'] = events_df['sri_centered'] / np.nanstd(events_df['sri_centered'])
    # events_df['sri_standardized'] = events_df['sri'] / np.nanstd(events_df['sri'])

    return events_df


def add_drop_log_to_events(
        epochs,
        events_df,
        drop_log_full,
        baseline_channels_dict,
        logger_preprocessing_info,
):
    channels = epochs.info['ch_names']
    channels_events_info = {}

    for channel in channels:
        channel_df = events_df.copy()

        baselines = baseline_channels_dict[channel] if channel in baseline_channels_dict else []
        channel_df['baseline'] = np.nan

        drops = drop_log_full[channel] if channel in drop_log_full else []
        channel_df['bad'] = np.nan

        time_0_indices = channel_df.index[channel_df['event'] == 'Time 0/'].tolist()

        # Assign drop log values to the corresponding Time 0/ indices
        for key, value in drops.items():
            if key < len(time_0_indices):
                channel_df.at[time_0_indices[key], 'bad'] = value

        # Assign baseline values to the corresponding Time 0/ indices
        for key, value in baselines.items():
            if key < len(time_0_indices):
                channel_df.at[time_0_indices[key], 'baseline'] = value

        # Propagate values to the matching latencies - Go event
        for idx in time_0_indices:
            matching_latency = channel_df.at[idx, 'latency']
            # Find the matching row where the event is not "Time 0/" but has the same latency
            matching_index = channel_df.index[
                (channel_df['latency'] == matching_latency) & (channel_df['event'] != 'Time 0/')].tolist()

            if matching_index:
                matching_index = matching_index[0]  # Assuming one matching row
                channel_df.at[matching_index, 'baseline'] = channel_df.at[idx, 'baseline']
                channel_df.at[matching_index, 'bad'] = channel_df.at[idx, 'bad']

        # Delete all "Time 0/" rows
        channel_df = channel_df[channel_df['event'] != 'Time 0/']

        # Propagate baseline values
        channel_df['baseline'] = channel_df['baseline'].ffill()
        # Propagate drop_log values
        channel_df['bad'] = channel_df['bad'].ffill()

        channels_events_info[channel] = channel_df

    return channels_events_info


def remove_incorrect_stop_responses(
        events_df,
        logger_preprocessing_info,
        target_sequence=None
):
    if target_sequence is None:
        target_sequence = ['go', 'response_stop', 'stop']
    matched_indices = []

    # Iterate through the 'type' column to find sequences
    for i in range(len(events_df) - len(target_sequence) + 1):
        if events_df['type'].iloc[i:i + len(target_sequence)].tolist() == target_sequence:
            matched_indices.extend(events_df.index[i:i + len(target_sequence)])
            logger_preprocessing_info.info(f'Response before STOP at {matched_indices}. Setting to BAD SEQ')

    events_df.loc[matched_indices, 'bad'] = 'BAD SEQ'

    return matched_indices, events_df


def transform_continuous_variables(events_df, columns=None, transforms=None, exclude_bad=True):
    if columns is None:
        raise ValueError("You must specify the variables (columns) to transform.")

    if transforms is None:
        transforms = [('center', lambda x: x - x.mean())]

    # Create a copy to avoid modifying the original DataFrame
    df_copy = events_df.copy()

    if exclude_bad:
        # Filter out rows where 'bad' is True
        df_copy = df_copy[df_copy['bad'] == False]

    for var in columns:
        for name, function in transforms:
            transformed_values = function(df_copy[var])
            transformed_series = pd.Series(index=events_df.index, dtype=float)
            transformed_series[df_copy.index] = transformed_values
            events_df[f'{var}_{name}'] = transformed_series

    return events_df


def preprocess(
        file_name,
        dir_name,
        picks,
        tmin,
        tmax,
        output_dir,
        logger_preprocessing_info,
        logger_errors_info,
        save_output=True,
        fill_missing=False,
        transforms=None,
        columns_to_transform=None
):
    epochs, events, event_id, sampling_rate = load_raw_brainvision(
        file_name=file_name,
        dir_name=dir_name,
        unified_events_dict=unified_events_dict,
        logger_preprocessing_info=logger_preprocessing_info,
        picks=picks,
        tmin=tmin,
        tmax=tmax,
    )

    events_df = create_events_dataframe(
        events,
        event_id,
        logger_preprocessing_info=logger_preprocessing_info,
        sampling_rate=sampling_rate,
        fill_missing=fill_missing
    )
    baseline_channels_dict, drop_log, drop_log_full = artifact_rejection(
        epochs,
        sampling_rate=sampling_rate,
        logger_preprocessing_info=logger_preprocessing_info
    )

    channels_events_info = add_drop_log_to_events(
        epochs,
        events_df,
        drop_log_full=drop_log_full,
        baseline_channels_dict=baseline_channels_dict,
        logger_preprocessing_info=logger_preprocessing_info,
    )

    for key, channel_events_info in channels_events_info.items():
        _, channel_events_info = remove_incorrect_stop_responses(
            channel_events_info,
            logger_preprocessing_info=logger_preprocessing_info
        )
        channels_events_info[key] = channel_events_info

    if (columns_to_transform is not None) and (transforms is not None):
        for key, channel_events_info in channels_events_info.items():
            channel_events_info = transform_continuous_variables(
                channel_events_info,
                columns=columns_to_transform,
                transforms=transforms,
                exclude_bad=True
            )
            channels_events_info[key] = channel_events_info

    # cast baseline from V to uV
    for key, channel_events_info in channels_events_info.items():
        channel_events_info['baseline'] = channel_events_info['baseline'] * 1000000
        channels_events_info[key] = channel_events_info

    # add event for baseline
    for key, channel_events_info in channels_events_info.items():
        go_rows = channel_events_info[channel_events_info['type'] == 'go'].copy()
        go_rows['type'] = 'baseline'

        channel_events_info = pd.concat([channel_events_info, go_rows], ignore_index=True)
        channel_events_info = channel_events_info.sort_values(by='latency').reset_index(drop=True)
        channels_events_info[key] = channel_events_info

    if save_output:
        output_file = f"{output_dir}/{file_name}"
        with open(f'{output_file}.pickle', 'wb') as handle:
            logger_preprocessing_info.info(f'Saving file {output_file}')
            pickle.dump(channels_events_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return channels_events_info, drop_log, epochs


def read_old_data(
        file_name,
        dir_name,
        tmin=-0.2,
        tmax=1.58,

):
    raw = read_raw_brainvision_local(f'../data/{dir_name}/{file_name}.vhdr', preload=True)
    events, event_id = mne.events_from_annotations(raw, unified_events_dict)

    epochs_old = mne.Epochs(
        raw,
        tmin=tmin,
        tmax=tmax,
        events=events,
        event_id=[0],
        baseline=None,
        preload=True,
    )

    return epochs_old


def plot_old_against_new(epochs_new, epochs_old, channels_of_interest, tmin, tmax, id):
    evokeds_new = epochs_new.average()
    evokeds_old = epochs_old.average()

    evokeds_new_data = evokeds_new.get_data(picks=channels_of_interest).mean(axis=0)
    evokeds_old_data = evokeds_old.get_data().flatten()

    x = np.linspace(tmin, tmax, len(evokeds_old_data))

    plt.figure()
    sns.lineplot(y=evokeds_new_data, x=x, color='r', label=f'New export: N = {epochs_new.get_data(copy=True).shape[0]}')
    sns.lineplot(y=evokeds_old_data, x=x, color='b', label=f'Old export: N = {epochs_old.get_data(copy=True).shape[0]}')
    plt.legend()

    # Add labels and legend
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (V)')
    plt.title(f"ID: {id}")
    plt.legend()
# %%
