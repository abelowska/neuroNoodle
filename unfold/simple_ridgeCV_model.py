import numpy as np
import pandas as pd
import logging

import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

# Import the Julia package manager
from juliacall import Pkg as jlPkg
from juliacall import Main as jl

from unfold import unfold_model

# Activate the environment in the current folder
jlPkg.activate(".")

# Check the status of the environment/installed packages -> will be empty at the beginning
print(jlPkg.status())

# Install Julia packages
jlPkg.add("Unfold")
jlPkg.add("DataFrames")
jlPkg.add(url="https://github.com/unfoldtoolbox/UnfoldDecode.jl")

jl.seval("using DataFrames")
jl.seval("using Unfold")
jl.seval("using UnfoldDecode")
Unfold = jl.Unfold
UnfoldDecode = jl.UnfoldDecode

logger = logging.getLogger(__name__)


def jl_results_to_python(results_jl):
    results_py_df = pd.DataFrame({
        'channel': results_jl.channel,
        'coefname': results_jl.coefname,
        'estimate': results_jl.estimate,
        'eventname': results_jl.eventname,
        'group': results_jl.group,
        'stderror': results_jl.stderror,
        'time': results_jl.time
    })
    return results_py_df


def evoked_from_raw(raw, events_df, event='go'):
    # Define the range of points to extract around each index
    start_offset = -13
    end_offset = 101

    events_indexes = events_df[events_df['type'] == event]['latency'].to_list()

    # Prepare a list to hold the 3D slices
    epochs = []

    # Loop through each index to extract the slices
    for index in events_indexes:
        # Calculate the start and end positions for the slice
        start = index + start_offset
        end = index + end_offset

        # Check if start and end indices are within bounds
        if start >= 0 and end <= raw.shape[-1]:
            # Extract the epoch with consistent shape
            epoch = raw[0, start:end]
            epochs.append(epoch)
        else:
            # Skip this index if it would result in an out-of-bounds slice
            continue

    evoked = np.mean(np.stack(epochs), axis=0)
    return evoked


def epochs_from_raw(raw, events_df, event='go', start_tp=-13, stop_tp=101):
    # Define the range of points to extract around each index
    start_offset = start_tp
    end_offset = stop_tp

    events_indexes = events_df[events_df['type'] == event]['latency'].to_list()

    # Prepare a list to hold the 3D slices
    epochs = []
    epochs_indices = []

    # Loop through each index to extract the slices
    for index in events_indexes:
        # Calculate the start and end positions for the slice
        start = index + start_offset
        end = index + end_offset
        epochs_indices.append((start, end))

        # Check if start and end indices are within bounds
        if start >= 0 and end <= raw.shape[-1]:
            # Extract the epoch with consistent shape
            epoch = raw[0, start:end]
            epochs.append(np.array(epoch))
        else:
            # Skip this index if it would result in an out-of-bounds slice
            continue

    epochs = np.stack(epochs)
    logger.debug(f"Epochs shape in epochs_from_raw: {epochs.shape}")
    return epochs, epochs_indices


def to_julia_vector(julia_type, python_list):
    return jl.seval("Vector{" + julia_type + "}")(python_list)


def get_events_jl(events_df):
    df = events_df

    df['sri'] = df['sri'].fillna(0)
    df['ssd'] = df['ssd'].fillna(0)

    type_column = to_julia_vector("String", df['type'].tolist())
    response_type_column = to_julia_vector("String", df['response_type'].tolist())
    stop_type_column = to_julia_vector("String", df['stop_type'].tolist())
    ssd_centered_column = to_julia_vector("Float64", df['ssd'].tolist())
    sri_centered_column = to_julia_vector(" Float64", df['sri'].tolist())
    latency_column = to_julia_vector("Int64", df['latency'].tolist())
    baseline_column = to_julia_vector("Float64", df['baseline'].tolist())

    # Create the Julia DataFrame
    events_df_jl = jl.DataFrame(
        type=type_column,
        latency=latency_column,
        ssd_centered=ssd_centered_column,
        sri_centered=sri_centered_column,
        response_type=response_type_column,
        stop_type=stop_type_column,
        baseline=baseline_column
    )
    return events_df_jl


class UnfoldSimpleRidgeCVModelPy:
    def __init__(self, u_model, linear_estimator):
        self.u_model = u_model
        self.num_parameters = None
        self.coeftable = None
        self.linear_estimator = linear_estimator
        self.num_parameters = None
        self.coefs = None,

    def fit(self, signal, events_df):
        model_matrix = self.get_model_matrix(signal, events_df)
        targets = np.ravel(signal)[: model_matrix.shape[0]]

        self.linear_estimator.fit(model_matrix, targets)

        self.num_parameters = model_matrix.shape[-1]
        self.coefs = self.linear_estimator.coef_

        coeftable = self.coeftable.copy()
        coeftable['estimate'] = self.coefs

        self.coeftable = coeftable

    def predict(self, signal, events_df):
        betas = self.coeftable.copy()['estimate'].to_numpy()

        signal = np.ravel(signal)
        events_df_jl = get_events_jl(events_df)

        # create X from events_df
        m = Unfold.fit(
            Unfold.UnfoldModel,
            self.u_model,
            events_df_jl,
            signal,
            eventcolumn="type",
        )
        X = Unfold.modelmatrix(m)

        signal_predicted = np.dot(X, betas).reshape(1, -1)

        return signal_predicted

    def get_model_matrix(self, signal, events_df):
        signal = np.ravel(signal)
        events_df_jl = unfold_model.get_events_jl(events_df)

        # Fit Unfold model
        m = Unfold.fit(
            Unfold.UnfoldModel,
            self.u_model,
            events_df_jl,
            signal,
            eventcolumn="type",
        )

        model_matrix = np.array(Unfold.modelmatrix(m))
        results_jl = Unfold.coeftable(m)
        self.coeftable = unfold_model.jl_results_to_python(results_jl)

        return model_matrix

    def get_results(self, predicted=True, signal=None, events_df=None):

        if predicted:
            predicted_signal = self.predict(signal, events_df)
            predicted_evoked = evoked_from_raw(predicted_signal, events_df, event='go')
            predicted_evoked_df = pd.DataFrame({
                'channel': 1,
                'coefname': 'predicted',
                'estimate': predicted_evoked,
                'eventname': 'evoked',
                'group': None,
                'stderror': None,
                'time': np.linspace(-0.2, 1.58, predicted_evoked.shape[-1])
            })

            original_evoked = evoked_from_raw(signal, events_df, event='go')
            original_evoked_df = pd.DataFrame({
                'channel': 1,
                'coefname': 'original',
                'estimate': original_evoked,
                'eventname': 'evoked',
                'group': None,
                'stderror': None,
                'time': np.linspace(-0.2, 1.58, original_evoked.shape[-1])
            })

            intercept_evoked_df = pd.DataFrame({
                'channel': 1,
                'coefname': '(Intercept)',
                'estimate': np.zeros(original_evoked.shape),
                'eventname': 'evoked',
                'group': None,
                'stderror': None,
                'time': np.linspace(-0.2, 1.58, original_evoked.shape[-1])
            })

            results_df = pd.concat([self.coeftable, predicted_evoked_df, original_evoked_df, intercept_evoked_df],
                                   axis=0,
                                   ignore_index=True)
            return results_df

        else:
            return self.coeftable


