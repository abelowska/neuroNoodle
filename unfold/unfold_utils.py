import mne
import numpy as np
import ipywidgets as widgets
from matplotlib import pyplot as plt


class EventData:
    def __init__(self, event_name):
        self.event_name = event_name
        self.channels = []
        self.times = None
        self.data = {}

    def add_channel_data(self, channel, estimates_by_coef, times):
        # Append the channel to the list
        self.channels.append(channel)

        # Set the time points if not already set
        if self.times is None:
            self.times = times

        # Append data for each coefname
        for coefname, estimates in estimates_by_coef.items():
            estimates_array = np.array(estimates)
            if coefname not in self.data:
                # Initialize with the first channel's data
                self.data[coefname] = estimates_array
            else:
                # Stack estimates across channels (along a new axis)
                self.data[coefname] = np.vstack([self.data[coefname], estimates_array])

    def get_channel_data(self, coefname):
        return self.data.get(coefname, None)


class ResultsContainer:
    def __init__(self, event_names):
        self.event_names = event_names
        self.events_data = {event_name: EventData(event_name) for event_name in event_names}

    def process_results(self, results_df):
        for event_name, event_data in self.events_data.items():
            for channel, unfold_data_df in results_df.items():
                # Filter events by event name
                data_df = unfold_data_df[unfold_data_df.eventname == event_name]

                # Group data by coefnames and collect estimates
                estimates_by_coef = data_df.groupby('coefname')['estimate'].apply(list).to_dict()

                # Add channel data to the event
                event_data.add_channel_data(channel, estimates_by_coef, data_df['time'].to_numpy())

    def get_event_data(self, event_name):
        return self.events_data.get(event_name, None)

    def get_all_event_names(self):
        return list(self.events_data.keys())


def create_evokeds_from_container(results_obj, sampling_rate, montage='biosemi32'):
    evoked_dict = {}

    # Iterate over each event in the ResultsContainer
    for event_name in results_obj.get_all_event_names():
        event_data = results_obj.get_event_data(event_name)

        # Extract channels and times from event_data object
        channels_list = event_data.channels
        times = event_data.times

        # Create MNE info object
        info = mne.create_info(ch_names=channels_list, sfreq=sampling_rate, ch_types='eeg')

        # Iterate over each coefname (e.g., (Intercept), or other coefficients)
        for coefname, data in event_data.data.items():
            # data is of shape (n_channels, n_times) for each channel
            # Create Evoked object directly from the data
            evoked_data = np.array(data)
            # cast to V from uV, for MNE
            evoked_data = evoked_data / 1000000

            # Create mne.EvokedArray, time should start from the first time point
            evoked = mne.EvokedArray(evoked_data, info=info, tmin=times[0])
            evoked.set_montage(montage)

            # Store the evoked object for this event_name and coefname
            evoked_dict[(event_name, coefname)] = evoked
    return evoked_dict


def average_evoked_across_ids(unfold_results_dict):
    # Initialize an empty dictionary to store the average results
    unfolded_results_dict_all = {'all': {}}

    # Get the list of all event_name, coef_name pairs
    all_event_coef_pairs = set()
    for evoked_dict in unfold_results_dict.values():
        all_event_coef_pairs.update(evoked_dict.keys())

    # Iterate over each (event_name, coef_name) pair
    for event_coef_pair in all_event_coef_pairs:
        evoked_list = []

        # Gather all the Evoked objects for this event_coef_pair from all ids
        for evoked_dict in unfold_results_dict.values():
            if event_coef_pair in evoked_dict:
                evoked_list.append(evoked_dict[event_coef_pair])

        # If we have more than one Evoked object, compute the grand average
        if len(evoked_list) > 0:
            evoked_avg = mne.grand_average(evoked_list)
            unfolded_results_dict_all['all'][event_coef_pair] = evoked_avg

    return unfolded_results_dict_all


def create_combined_evoked(event_name, coef_dict, evoked_dict):
    """
    Create two evoked objects based on the specified event name and coefficient multipliers.

    Parameters:
    - event_name: str, the name of the event to process.
    - coef_dict: dict, where keys are coefnames and values are either False or a tuple with multipliers.
    - evoked_dict: dict, where keys are tuples (event_name, coefname) and values are mne Evoked objects.

    Returns:
    - evoked1: mne Evoked object, for Intercept + multiplier1 * (event_name, coefname)
    - evoked2: mne Evoked object, for Intercept + multiplier2 * (event_name, coefname)
    """
    # Retrieve the Intercept evoked
    intercept_key = (event_name, '(Intercept)')

    if intercept_key not in evoked_dict:
        raise ValueError(f"No evoked found for key: {intercept_key}")

    intercept_evoked = evoked_dict[intercept_key]

    # Initialize evoked objects for the new combinations
    evoked1 = intercept_evoked.copy()
    evoked2 = intercept_evoked.copy()

    for coefname, multipliers in coef_dict.items():
        if multipliers is False:
            continue  # Skip this coefname

        # Get the evoked for the current coefname
        multiplier_lower, multiplier_upper = multipliers
        coef_key = (event_name, coefname)

        if coef_key in evoked_dict:
            coef_evoked = evoked_dict[coef_key]

            # Calculate new evoked data

            # cast to uV (as it was fitted) to ensure compatibility
            evoked1.data = evoked1.data * 1000000
            evoked2.data = evoked2.data * 1000000

            evoked1.data += multiplier_lower * (coef_evoked.data * 1000000)
            evoked2.data += multiplier_upper * (coef_evoked.data * 1000000)

            # re-cast to V
            evoked1.data = evoked1.data / 1000000
            evoked2.data = evoked2.data / 1000000

        else:
            print(f"Warning: No evoked found for key: {coef_key}")

    return evoked1, evoked2


def plot_evoked_dict(evoked_dict):
    event_names = sorted(set(event for event, _ in evoked_dict.keys()))  # Get unique event names

    # Create a dropdown for event names
    event_dropdown = widgets.Dropdown(
        options=event_names,
        description='Event Name:',
        value='baseline'
    )

    # Create a toggle for electrodes
    electrode_toggle = widgets.Dropdown(
        options=evoked_dict[('baseline', '(Intercept)')].info['ch_names'],  # Add your electrodes here
        description='Electrode:',
        value='Cz'
    )

    output = widgets.Output()

    # Global container for min and max multiplier text boxes
    min_multiplier_boxes = []
    max_multiplier_boxes = []
    coef_checkboxes = []

    # Slider for selecting time for topomap
    time_slider = widgets.FloatSlider(
        description='Time (s)',
        continuous_update=False,  # Only update when user releases the slider
        layout=widgets.Layout(width='1070px')
    )

    def update_coef_checkboxes(event_name):
        global min_multiplier_boxes, max_multiplier_boxes, coef_checkboxes
        # Extract coefficient names based on the selected event, excluding '(Intercept)'
        coefnames = [coef for event, coef in evoked_dict.keys() if event == event_name and coef != '(Intercept)']

        # Create checkboxes for coefnames
        coef_checkboxes = [widgets.Checkbox(value=False, description=name) for name in set(coefnames)]

        # Reset the min and max multiplier boxes
        min_multiplier_boxes = []
        max_multiplier_boxes = []

        # Create a list of HBox rows where each row contains a checkbox and its corresponding multiplier text boxes
        rows = []
        for i, name in enumerate(set(coefnames)):
            min_box = widgets.FloatText(value=100, description='Min:', layout=widgets.Layout(width='150px'))
            max_box = widgets.FloatText(value=500, description='Max:', layout=widgets.Layout(width='150px'))
            min_multiplier_boxes.append(min_box)
            max_multiplier_boxes.append(max_box)
            # Create a horizontal row of checkbox + min + max
            row = widgets.HBox([coef_checkboxes[i], min_box, max_box])
            rows.append(row)

        # If there are no coefficients, return just the checkboxes (empty in case of only Intercept)
        if not rows:
            return widgets.VBox(coef_checkboxes)

        # Otherwise return the VBox containing all the rows
        return widgets.VBox(rows)

    def update_plot(event_name, coef_selection, selected_electrode, selected_time):
        # Clear previous output
        with output:
            output.clear_output()

            # Prepare coef_dict based on checked boxes, always including '(Intercept)'
            coef_dict = {'(Intercept)': (1, 1)}  # Default coefficient values

            # Iterate through the selected coefficients and apply the corresponding multipliers
            for i, row in enumerate(coef_selection):
                checkbox = row.children[0]  # The checkbox is the first element in the HBox
                if checkbox.value:  # If the checkbox is selected
                    coefname = checkbox.description
                    coef_dict[coefname] = (min_multiplier_boxes[i].value, max_multiplier_boxes[i].value)

            # Generate combined evoked based on user selection
            evoked1, evoked2 = create_combined_evoked(event_name, coef_dict, evoked_dict)

            # Update the slider range based on evoked1 times
            time_slider.min = evoked1.times[0]
            time_slider.max = evoked1.times[-1]
            time_slider.step = evoked1.times[1] - evoked1.times[0]  # Use the time step of the data
            if time_slider.value < time_slider.min or time_slider.value > time_slider.max:
                time_slider.value = 0.2  # Set default if it's out of range

            # Get the data for the selected electrode
            ch_idx = evoked1.ch_names.index(selected_electrode)  # Find index of selected electrode

            # Plot the line plot for the selected electrode
            plt.figure(figsize=(12, 6))
            plt.plot(evoked1.times, evoked1.data[ch_idx], label='Intercept + Min Multiplier', color='blue')
            plt.plot(evoked2.times, evoked2.data[ch_idx], label='Intercept + Max Multiplier', color='red')
            plt.axvline(x=0, color='gray', linestyle='--', label=f'Time: 0s')  # Example vertical line at time = 0
            plt.title(f'Evoked Response at {selected_electrode}')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude (uV)')
            plt.legend()
            plt.grid()
            plt.show()

            # Display the slider below the main plot
            display(time_slider)

            mne.viz.plot_evoked_topomap(evoked1, times=selected_time, show=False)
            plt.suptitle(f"Intercept + Min Multiplier at time {selected_time:.3f}s")
            plt.show()

            mne.viz.plot_evoked_topomap(evoked2, times=selected_time, show=False)
            plt.suptitle(f"Intercept + Max Multiplier at time {selected_time:.3f}s")
            plt.show()

    # Attach listeners to the checkboxes and multiplier text boxes
    def attach_listeners():
        for i, checkbox in enumerate(coef_checkboxes):
            checkbox.observe(lambda change, idx=i: update_plot(event_dropdown.value, coef_checkboxes_widget.children,
                                                               electrode_toggle.value, time_slider.value),
                             names='value')
            min_multiplier_boxes[i].observe(
                lambda change, idx=i: update_plot(event_dropdown.value, coef_checkboxes_widget.children,
                                                  electrode_toggle.value, time_slider.value), names='value')
            max_multiplier_boxes[i].observe(
                lambda change, idx=i: update_plot(event_dropdown.value, coef_checkboxes_widget.children,
                                                  electrode_toggle.value, time_slider.value), names='value')

    # Add a listener to the slider
    time_slider.observe(
        lambda change: update_plot(event_dropdown.value, coef_checkboxes_widget.children, electrode_toggle.value,
                                   time_slider.value), names='value')

    # Link the interactive widgets to the update function
    def update_all(*args):
        # Update coefficient checkboxes and text boxes when the event name changes
        coef_checkboxes_widget.children = update_coef_checkboxes(event_dropdown.value).children
        # Attach listeners to the new checkboxes and text boxes
        attach_listeners()
        # Update plot after changing event
        update_plot(event_dropdown.value, coef_checkboxes_widget.children, electrode_toggle.value, time_slider.value)

    # Create the initial checkboxes for the default event

    coef_checkboxes_widget = update_coef_checkboxes(event_dropdown.value)

    # Attach the event listener to the electrode toggle
    electrode_toggle.observe(
        lambda change: update_plot(event_dropdown.value, coef_checkboxes_widget.children, electrode_toggle.value,
                                   time_slider.value), names='value')

    # Attach the event listener to the event dropdown
    event_dropdown.observe(lambda change: update_all(), names='value')

    # Display the widgets and output area
    display(event_dropdown, coef_checkboxes_widget, electrode_toggle, output)

    # Call the update function initially
    update_all()

def plot_results_dict(unfold_results_dict):
    # Create a dropdown for selecting the ID from unfold_results_dict
    id_dropdown = widgets.Dropdown(
        options=list(unfold_results_dict.keys()),
        description='Select ID:',
        value=list(unfold_results_dict.keys())[0]  # Default to the first id
    )
    
    # Get the corresponding evoked_dict for the initially selected ID
    evoked_dict = unfold_results_dict[id_dropdown.value]

    # Create a dropdown for event names
    event_names = sorted(set(event for event, _ in evoked_dict.keys()))  # Get unique event names

    event_dropdown = widgets.Dropdown(
        options=event_names,
        description='Event Name:',
        value='baseline'
    )

    # Create a toggle for electrodes
    electrode_toggle = widgets.Dropdown(
        options=evoked_dict[('baseline', '(Intercept)')].info['ch_names'],  # Add your electrodes here
        description='Electrode:',
        value='Cz'
    )

    output = widgets.Output()

    # Local containers for min and max multiplier text boxes and checkboxes
    min_multiplier_boxes = []
    max_multiplier_boxes = []
    coef_checkboxes = []

    # Slider for selecting time for topomap
    time_slider = widgets.FloatSlider(
        description='Time (s)',
        continuous_update=False,  # Only update when user releases the slider
        layout=widgets.Layout(width='1070px')
    )

    def update_coef_checkboxes(event_name):
        nonlocal min_multiplier_boxes, max_multiplier_boxes, coef_checkboxes
        # Extract coefficient names based on the selected event, excluding '(Intercept)'
        coefnames = [coef for event, coef in evoked_dict.keys() if event == event_name and coef != '(Intercept)']

        # Reset and create new checkboxes for coefnames
        coef_checkboxes = [widgets.Checkbox(value=False, description=name) for name in set(coefnames)]

        # Reset the min and max multiplier boxes
        min_multiplier_boxes = []
        max_multiplier_boxes = []

        # Create a list of HBox rows where each row contains a checkbox and its corresponding multiplier text boxes
        rows = []
        for i, name in enumerate(set(coefnames)):
            min_box = widgets.FloatText(value=100, description='Min:', layout=widgets.Layout(width='150px'))
            max_box = widgets.FloatText(value=500, description='Max:', layout=widgets.Layout(width='150px'))
            min_multiplier_boxes.append(min_box)
            max_multiplier_boxes.append(max_box)
            # Create a horizontal row of checkbox + min + max
            row = widgets.HBox([coef_checkboxes[i], min_box, max_box])
            rows.append(row)

        # If there are no coefficients, return just the checkboxes (empty in case of only Intercept)
        if not rows:
            return widgets.VBox(coef_checkboxes)

        # Otherwise return the VBox containing all the rows
        return widgets.VBox(rows)

    def update_plot(event_name, coef_selection, selected_electrode, selected_time):
        with output:
            output.clear_output()

            # Prepare coef_dict based on checked boxes, always including '(Intercept)'
            coef_dict = {'(Intercept)': (1, 1)}  # Default coefficient values

            # Iterate through the selected coefficients and apply the corresponding multipliers
            for i, row in enumerate(coef_selection):
                checkbox = row.children[0]  # The checkbox is the first element in the HBox
                if checkbox.value:  # If the checkbox is selected
                    coefname = checkbox.description
                    coef_dict[coefname] = (min_multiplier_boxes[i].value, max_multiplier_boxes[i].value)

            # Generate combined evoked based on user selection
            evoked1, evoked2 = create_combined_evoked(event_name, coef_dict, evoked_dict)

            # Update the slider range based on evoked1 times
            time_slider.min = evoked1.times[0]
            time_slider.max = evoked1.times[-1]
            time_slider.step = evoked1.times[1] - evoked1.times[0]
            if time_slider.value < time_slider.min or time_slider.value > time_slider.max:
                time_slider.value = 0.2

            # Get the data for the selected electrode
            ch_idx = evoked1.ch_names.index(selected_electrode)

            # Plot the line plot for the selected electrode
            plt.figure(figsize=(12, 6))
            plt.plot(evoked1.times, evoked1.data[ch_idx], label='Intercept + Min Multiplier', color='blue')
            plt.plot(evoked2.times, evoked2.data[ch_idx], label='Intercept + Max Multiplier', color='red')
            plt.axvline(x=0, color='gray', linestyle='--')
            plt.title(f'Evoked Response at {selected_electrode}')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude (uV)')
            plt.legend()
            plt.grid()
            plt.show()

            display(time_slider)

            mne.viz.plot_evoked_topomap(evoked1, times=selected_time, show=False)
            plt.suptitle(f"Intercept - Min Multiplier at time {selected_time:.3f}s")
            plt.show()

            mne.viz.plot_evoked_topomap(evoked2, times=selected_time, show=False)
            plt.suptitle(f"Intercept + Max Multiplier at time {selected_time:.3f}s")
            plt.show()

    def attach_listeners():
        for i, checkbox in enumerate(coef_checkboxes):
            checkbox.observe(lambda change, idx=i: update_plot(event_dropdown.value, coef_checkboxes_widget.children,
                                                               electrode_toggle.value, time_slider.value),
                             names='value')
            min_multiplier_boxes[i].observe(
                lambda change, idx=i: update_plot(event_dropdown.value, coef_checkboxes_widget.children,
                                                  electrode_toggle.value, time_slider.value), names='value')
            max_multiplier_boxes[i].observe(
                lambda change, idx=i: update_plot(event_dropdown.value, coef_checkboxes_widget.children,
                                                  electrode_toggle.value, time_slider.value), names='value')

    # Add a listener to the slider
    time_slider.observe(
        lambda change: update_plot(event_dropdown.value, coef_checkboxes_widget.children, electrode_toggle.value,
                                   time_slider.value), names='value')

    def update_evoked_dict(*args):
        nonlocal evoked_dict
        evoked_dict = unfold_results_dict[id_dropdown.value]
        event_names = sorted(set(event for event, _ in evoked_dict.keys()))
        event_dropdown.options = event_names
        event_dropdown.value = 'baseline'
        electrode_toggle.options = evoked_dict[('baseline', '(Intercept)')].info['ch_names']
        electrode_toggle.value = 'Cz'
        update_all()

    def update_all(*args):
        coef_checkboxes_widget.children = update_coef_checkboxes(event_dropdown.value).children
        attach_listeners()
        update_plot(event_dropdown.value, coef_checkboxes_widget.children, electrode_toggle.value, time_slider.value)

    coef_checkboxes_widget = update_coef_checkboxes(event_dropdown.value)
    electrode_toggle.observe(lambda change: update_plot(event_dropdown.value, coef_checkboxes_widget.children,
                                                        electrode_toggle.value, time_slider.value), names='value')
    event_dropdown.observe(lambda change: update_all(), names='value')
    id_dropdown.observe(update_evoked_dict, names='value')

    display(id_dropdown, event_dropdown, coef_checkboxes_widget, electrode_toggle, output)

    update_all()
