import numpy as np
import pandas as pd


def gaussian_peak(t, center, width, amplitude):
    """
    Generate a Gaussian peak.

    Parameters:
    t (np.ndarray): Time array.
    center (float): Center of the peak.
    width (float): Width (standard deviation) of the Gaussian peak.
    amplitude (float): Amplitude of the peak.

    Returns:
    np.ndarray: Gaussian peak values.
    """
    return amplitude * np.exp(-((t - center) ** 2) / (2 * width ** 2))


def simulate_erp_components_with_conditions(
        n_trials_per_ssd,
        ssd_list,
        config,
        sampling_rate=1000,
        noise_level=0.1,
        duration=1.2,
):
    """
    Simulate ERP components with specific conditions.

    Parameters:
    n_trials_per_ssd (int): Number of trials for SSD value.
    distances_C_A (list): List of distances between C and A in seconds.
    sampling_rate (int): Sampling rate in Hz.
    noise_level (float): Standard deviation of the added Gaussian noise.

    Returns:
    np.ndarray: Array of simulated ERP trials of shape (n_trials, n_samples).
    """
    # Parameters
    duration = duration  # Duration of the signal in seconds
    t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)

    # Parameters for peaks
    peak_center_go = 0.2  # Center of Component C peak in seconds
    peak_width_go = 0.1  # Width of Component C peak in seconds
    amplitude_go_intercept = config['go_intercept']  # Amplitude of Component C peak

    peak_width_stop = 0.1  # Width of Component A peak in seconds
    peak_width_response = 0.1  # Width of Component B peak in seconds

    # Initialize list to hold ERP trials
    erp_trials = []
    data_df = pd.DataFrame()

    # Generate ERP trials
    for ssd in ssd_list:
        for _ in range(n_trials_per_ssd):
            trial_ssd = ssd
            sri = np.nan
            rt = np.nan
            # Simulate Component C
            amplitude_go = amplitude_go_intercept
            latency_go = int((peak_center_go - peak_width_go) * sampling_rate)
            component_go = gaussian_peak(t, peak_center_go, peak_width_go, amplitude_go)

            # Determine if Component STOP is generated (30% of the time)
            if np.random.rand() < config['probability']['stop']:
                # Component STOP is generated
                peak_center_stop = peak_center_go + trial_ssd

                if config['interactions']['stop x SSD']:
                    amplitude_stop = config['stop_intercept'] + trial_ssd * 1.2
                else:
                    amplitude_stop = config['stop_intercept']
                latency_stop = int((peak_center_stop - peak_width_stop) * sampling_rate)
                stop_type = 'stop'
                component_stop = gaussian_peak(t, peak_center_stop, peak_width_stop, amplitude_stop)

                # Determine if Component B is 'error' (30%) or 'correct' (70%)
                if np.random.rand() < config['probability']['error']:
                    # 'Error' B
                    sri = np.random.normal(config['sri']['loc'], config['sri']['scale'])
                    rt = trial_ssd + sri + (peak_center_go - peak_width_go)
                    amplitude_response_intercept = config['response_intercept']['error']
                    amplitude_response = amplitude_response_intercept

                    if config['interactions']['response_error x SRI']:
                        amplitude_response = amplitude_response + sri * 1.5  # Amplitude increases with distance
                    if config['interactions']['response_error x SSD']:
                        amplitude_response = amplitude_response + trial_ssd * 1.2

                    peak_center_response = peak_center_stop + sri
                    # peak_center_response = peak_center_stop
                    latency_response = int((peak_center_response - peak_width_response) * sampling_rate)
                    response_type = 'error'
                    component_response = gaussian_peak(t, peak_center_response, peak_width_response, amplitude_response)

                else:
                    latency_response = np.nan
                    amplitude_response = np.nan
                    response_type = 'correct'
                    component_response = np.zeros_like(t)

            else:
                # Component A is not generated
                latency_stop = np.nan
                amplitude_stop = np.nan
                stop_type = 'nostop'
                trial_ssd = np.nan
                component_stop = np.zeros_like(t)

                # Component B is generated at a random distance after C
                rt = np.random.normal(0.5, 0.15)
                # rt = 0.4
                peak_center_response = peak_center_go + rt

                amplitude_response_intercept = config['response_intercept']['correct_nostop']  # Constant amplitude
                amplitude_response = amplitude_response_intercept
                latency_response = int((peak_center_response - peak_width_response) * sampling_rate)
                response_type = 'nostop_correct'

                component_response = gaussian_peak(t, peak_center_response, peak_width_response, amplitude_response)

            # Combine components
            erp_trial = component_go + component_stop + component_response

            # Add Gaussian noise to the ERP trial
            noise = np.random.normal(0, noise_level, erp_trial.shape)
            erp_trial += noise

            erp_trials.append(erp_trial)

            this_df = pd.DataFrame({
                'signal': [np.array(erp_trial)],
                'SSD': [trial_ssd],
                'SRI': [sri],
                'Go amplitude': [amplitude_go],
                'Stop amplitude': [amplitude_stop],
                'Response amplitude': [amplitude_response],
                'Go event latency': [latency_go],
                'Stop event latency': [latency_stop],
                'Response event latency': [latency_response],
                'stop_type': [stop_type],
                'response_type': [response_type],
                'rt': [rt],
            })

            data_df = pd.concat([data_df, this_df], ignore_index=True)

            # Convert list to numpy array
    erp_trials = np.array(erp_trials)

    return erp_trials, data_df


def simulate_erp_components_with_conditions_test(
        n_trials_per_ssd,
        ssd_list,
        config,
        sampling_rate=1000,
        noise_level=0.1,
        duration=1.2,
        stop_diff=True,
        remove_ern=False,
        stop_component_offset=0.25
):
    """
    Simulate ERP components with specific conditions.

    Parameters:
    n_trials_per_ssd (int): Number of trials for SSD value.
    distances_C_A (list): List of distances between C and A in seconds.
    sampling_rate (int): Sampling rate in Hz.
    noise_level (float): Standard deviation of the added Gaussian noise.

    Returns:
    np.ndarray: Array of simulated ERP trials of shape (n_trials, n_samples).
    """
    # Parameters
    duration = duration  # Duration of the signal in seconds
    t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)

    # Parameters for peaks
    peak_center_go = 0.2  # Center of Component C peak in seconds
    peak_width_go = 0.1  # Width of Component C peak in seconds
    peak_width_stop = 0.1  # Width of Component STOP peak in seconds
    peak_width_response = 0.1  # Width of Component RESPONSE peak in seconds

    # Initialize list to hold ERP trials
    erp_trials = []
    data_df = pd.DataFrame()

    # Generate ERP trials
    for ssd in ssd_list:
        for _ in range(n_trials_per_ssd):

            stop = True if np.random.rand() < config['probability']['stop'] else False
            response_stop = True if np.random.rand() < config['probability']['error'] else False

            # Simulate Component GO
            amplitude_go_intercept = config['go_intercept']  # Amplitude of Component C peak
            amplitude_go = amplitude_go_intercept
            latency_go = int((peak_center_go - peak_width_go) * sampling_rate)
            component_go = gaussian_peak(t, peak_center_go, peak_width_go, amplitude_go)

            # Simulate Component STOP
            if stop is True:
                trial_ssd = ssd
                peak_center_stop = peak_center_go + trial_ssd + stop_component_offset

                if response_stop is True:
                    amplitude_stop_intercept = config['stop_intercept'] + 0.3 if stop_diff else config['stop_intercept']
                else:
                    amplitude_stop_intercept = config['stop_intercept']

                if config['interactions']['stop x SSD']:
                    amplitude_stop = amplitude_stop_intercept + trial_ssd * 1.2
                else:
                    amplitude_stop = amplitude_stop_intercept

                latency_stop = int((peak_center_go + trial_ssd - peak_width_stop) * sampling_rate)
                stop_type = 'stop'
                component_stop = gaussian_peak(t, peak_center_stop, peak_width_stop, amplitude_stop)

                if response_stop is True:
                    # 'Error'
                    sri = np.random.normal(config['sri']['loc'], config['sri']['scale'])
                    rt = trial_ssd + sri + (peak_center_go - peak_width_go)
                    amplitude_response_intercept = config['response_intercept']['error']
                    amplitude_response = amplitude_response_intercept

                    if config['interactions']['response_error x SRI']:
                        amplitude_response = amplitude_response + sri * 1.5  # Amplitude increases with distance
                    if config['interactions']['response_error x SSD']:
                        amplitude_response = amplitude_response + trial_ssd * 1.2

                    peak_center_response = peak_center_go + trial_ssd + sri
                    latency_response = int((peak_center_go + trial_ssd + sri - peak_width_response) * sampling_rate)
                    response_type = 'error'
                    if remove_ern is False:
                        component_response = gaussian_peak(t, peak_center_response, peak_width_response,
                                                           amplitude_response)
                    else:
                        component_response = np.zeros_like(t)
                else:
                    # 'Correct inhibited'
                    latency_response = np.nan
                    amplitude_response = np.nan
                    peak_center_response = np.nan
                    sri = np.nan
                    rt = np.nan
                    response_type = 'correct'
                    component_response = np.zeros_like(t)

            # Simulate Component RESPONSE-NOSTOP
            else:
                # Component STOP is not generated
                amplitude_stop = np.nan
                latency_stop = np.nan
                trial_ssd = np.nan
                sri = np.nan
                stop_type = 'nostop'
                peak_center_stop = np.nan
                component_stop = np.zeros_like(t)

                # Component RESPONSE-NOSTOP is generated at a random distance after C
                rt = np.random.normal(0.5, 0.15)
                peak_center_response = peak_center_go + rt

                amplitude_response_intercept = config['response_intercept']['correct_nostop']  # Constant amplitude
                amplitude_response = amplitude_response_intercept
                latency_response = int((peak_center_go + rt - peak_width_response) * sampling_rate)
                response_type = 'nostop_correct'

                component_response = gaussian_peak(t, peak_center_response, peak_width_response, amplitude_response)

            # Combine components
            erp_trial = component_go + component_stop + component_response

            # Add Gaussian noise to the ERP trial
            noise = np.random.normal(0, noise_level, erp_trial.shape)
            erp_trial += noise

            erp_trials.append(erp_trial)

            this_df = pd.DataFrame({
                'signal': [np.array(erp_trial)],
                'SSD': [trial_ssd],
                'SRI': [sri],
                'Go amplitude': [amplitude_go],
                'Stop amplitude': [amplitude_stop],
                'Response amplitude': [amplitude_response],
                'Go event latency': [latency_go],
                'Stop event latency': [latency_stop],
                'Response event latency': [latency_response],
                'stop_type': [stop_type],
                'response_type': [response_type],
                'rt': [rt],
                'peak_center_go': [peak_center_go],
                'peak_center_stop': [peak_center_stop],
                'peak_center_response': [peak_center_response],
            })

            data_df = pd.concat([data_df, this_df], ignore_index=True)

            # Convert list to numpy array
    erp_trials = np.array(erp_trials)

    return erp_trials, data_df


def create_events_table(data_df, sampling_rate, duration):
    rows = []

    # Iterate through the input DataFrame and populate the list
    for idx, row in data_df.iterrows():
        signal_offset = idx * int(duration * sampling_rate)

        # Add each event to the list with the appropriate latency and event type
        rows.append({
            'latency': signal_offset + row['Go event latency'],
            'event': 'go',
            'SSD': row['SSD'] if pd.notna(row['Go event latency']) else np.nan,
            'SRI': row['SRI'] if pd.notna(row['Go event latency']) else np.nan,
            'stop_type': 'SS' if (row['stop_type'] == 'stop') and (row['response_type'] == 'correct')
            else 'SU' if (row['stop_type'] == 'stop') and (row['response_type'] == 'error')
            else 'n-a',  # Default to NaN or another value if neither condition is met
            'response_type': row['response_type']
        })

        if pd.notna(row['Stop event latency']):
            rows.append({
                'latency': signal_offset + row['Stop event latency'],
                'event': 'stop',
                'SSD': row['SSD'],
                'SRI': row['SRI'] if pd.notna(row['Response event latency']) else np.nan,
                'stop_type': 'SS' if (row['stop_type'] == 'stop') and (row['response_type'] == 'correct')
                else 'SU' if (row['stop_type'] == 'stop') and (row['response_type'] == 'error')
                else 'n-a',  # Default to NaN or another value if neither condition is met
                'response_type': row['response_type']
            })

            if pd.notna(row['Response event latency']):
                rows.append({
                    'latency': signal_offset + row['Response event latency'],
                    'event': 'response',
                    'SSD': row['SSD'],
                    'SRI': row['SRI'],
                    'stop_type': 'SS' if (row['stop_type'] == 'stop') and (row['response_type'] == 'correct')
                    else 'SU' if (row['stop_type'] == 'stop') and (row['response_type'] == 'error')
                    else 'n-a',  # Default to NaN or another value if neither condition is met
                    'response_type': row['response_type']
                })
        else:
            rows.append({
                'latency': signal_offset + row['Response event latency'],
                'event': 'response_nostop',
                'SSD': np.nan,
                'SRI': np.nan,
                'stop_type': 'SS' if (row['stop_type'] == 'stop') and (row['response_type'] == 'correct')
                else 'SU' if (row['stop_type'] == 'stop') and (row['response_type'] == 'error')
                else 'n-a',  # Default to NaN or another value if neither condition is met
                'response_type': row['response_type']
            })

            # Create a new DataFrame from the list
    events_df = pd.DataFrame(rows)

    return events_df
