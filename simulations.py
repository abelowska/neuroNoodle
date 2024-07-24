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
                    amplitude_stop = config['stop_intercept'] + (trial_ssd - 0.1) * 1.2
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
                        amplitude_response = amplitude_response + (trial_ssd - 0.1) * 1.2

                    peak_center_response = peak_center_stop + sri
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
