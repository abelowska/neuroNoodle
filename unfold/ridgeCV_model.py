import numpy as np
import pandas as pd
from himalaya.ridge import RidgeCV
import logging
import sys

sys.path.insert(1, '../')

import unfold_model

# Import the Julia package manager
from juliacall import Pkg as jlPkg
from juliacall import Main as jl

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


def get_epochs_indices(events_df, event='go', start_tp=-13, stop_tp=101):
    # Define the range of points to extract around each index
    start_offset = start_tp
    end_offset = stop_tp

    events_indexes = events_df[events_df['type'] == event]['latency'].to_list()

    epochs_indices = []

    # Loop through each index to extract the slices
    for index in events_indexes:
        # Calculate the start and end positions for the slice
        start = index + start_offset
        end = index + end_offset
        epochs_indices.append((start, end))

    return epochs_indices


def epochs_from_raw(raw, epochs_indices):
    # Prepare a list to hold the 3D slices
    epochs = []

    # Loop through each index to extract the slices
    for start, end in epochs_indices:
        # Check if start and end indices are within bounds
        if start >= 0 and end <= raw.shape[-1]:
            # Extract the epoch with consistent shape
            epoch = raw[0, start:end]
            epochs.append(np.array(epoch))
        else:
            # Skip this index if it would result in an out-of-bounds slice
            continue

    epochs = np.stack(epochs)
    return epochs


def epochs_from_model_matrix(model_matrix, epochs_indices):
    model_matrix_epochs = []
    for start, end in epochs_indices:
        if start >= 0 and end <= model_matrix.shape[0]:
            # Extract the epoch with consistent shape
            model_matrix_epoch = model_matrix[start:end, :]
            model_matrix_epochs.append(np.array(model_matrix_epoch))
        else:
            # Skip this index if it would result in an out-of-bounds slice
            continue

    model_matrix_epochs = np.stack(model_matrix_epochs)
    return model_matrix_epochs


class UnfoldRidgeCVPy:
    def __init__(self, u_model, linear_estimator=RidgeCV()):
        self.u_model = u_model
        self.linear_estimator = linear_estimator
        self.num_parameters = None
        self.coeftable = None
        self.coefs = None

    def fit(self, signal, events_df):
        model_matrix = self.get_model_matrix(signal, events_df)
        epochs_indices = get_epochs_indices(events_df)

        eeg_epochs = epochs_from_raw(signal, epochs_indices)
        model_matrix_epochs = epochs_from_model_matrix(model_matrix,
                                                       epochs_indices)  # shape of (n_samples, n_targets, n_predictors)

        targets = eeg_epochs  # shape of (n_samples, n_targets) -> n_epochs, epoch_len
        X = model_matrix_epochs.reshape(eeg_epochs.shape[0], -1)  # shape of (n_samples, n_targets * n_predictors)

        self.linear_estimator.fit(X, targets)

        self.num_parameters = X.shape[-1]
        coefs = self.linear_estimator.coef_
        self.coefs = coefs

    def predict(self, signal, events_df):
        betas = self.coefs

        model_matrix = self.get_model_matrix(signal, events_df)
        epochs_indices = get_epochs_indices(events_df)

        eeg_epochs = epochs_from_raw(signal, epochs_indices)
        model_matrix_epochs = epochs_from_model_matrix(model_matrix, epochs_indices)

        X = model_matrix_epochs.reshape(eeg_epochs.shape[0], -1)  # shape of (n_samples, n_targets * n_predictors)

        # shape of (n_samples, n_targets)
        signal_predicted = np.dot(X, betas)

        return signal_predicted

    def get_results(self, predicted=True, signal=None, events_df=None):
        return

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

# class UnfoldRidgeCVSeparateTargetsPy:
#     def __init__(self, u_model, linear_estimator=RidgeCV()):
#         self.u_model = u_model
#         self.linear_estimator = linear_estimator
#         self.num_parameters = None
#         self.coeftable = None
#         self.coefs = None
#
#     def fit(self, signal, events_df):
#         model_matrix = self.get_model_matrix(signal, events_df)
#         epochs_indices = get_epochs_indices(events_df)
#
#         eeg_epochs = epochs_from_raw(signal, epochs_indices)
#         model_matrix_epochs = epochs_from_model_matrix(model_matrix, epochs_indices) # shape of (n_samples, n_targets, n_predictors)
#
#         targets = eeg_epochs  # shape of (n_samples, n_targets) -> n_epochs, epoch_len
#         X = model_matrix_epochs
#
#         betas = []
#         alphas = []
#
#         for i in range(0, targets.shape[-1]):
#             target = targets[:,i]
#             data = X[:, i, :]
#
#             self.linear_estimator.fit(data, target)
#
#             betas.append(self.linear_estimator.coef_)
#             alphas.append(self.linear_estimator.best_alphas_)
#
#         self.coefs = betas
#
#     def predict(self, signal, events_df):
#         betas = self.coefs
#
#         model_matrix = self.get_model_matrix(signal, events_df)
#         epochs_indices = get_epochs_indices(events_df)
#
#         eeg_epochs = epochs_from_raw(signal, epochs_indices)
#         model_matrix_epochs = epochs_from_model_matrix(model_matrix, epochs_indices)
#
#         # shape of (n_samples, n_targets)
#         signal_predicted = np.dot(X, betas)
#
#         return signal_predicted
#
#     def get_results(self, predicted=True, signal=None, events_df=None):
#         return
#
#     def get_model_matrix(self, signal, events_df):
#         signal = np.ravel(signal)
#         events_df_jl = unfold_model.get_events_jl(events_df)
#
#         # Fit Unfold model
#         m = Unfold.fit(
#             Unfold.UnfoldModel,
#             self.u_model,
#             events_df_jl,
#             signal,
#             eventcolumn="type",
#         )
#
#         model_matrix = np.array(Unfold.modelmatrix(m))
#         results_jl = Unfold.coeftable(m)
#         self.coeftable = unfold_model.jl_results_to_python(results_jl)
#
#         return model_matrix
