import numpy as np
import pandas as pd

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


def evoked_from_raw(raw, events_df, event='go'):
    # Define the range of points to extract around each index
    start_offset = -13
    end_offset = 101

    events_indexes = events_df[events_df['type'] == event]['latency'].to_list()

    # Prepare a list to hold the 3D slices
    epochs = []

    # Loop through each index to extract the slices
    for index in events_indexes:
        if index + end_offset < raw.shape[1]:
            # Calculate the start and end positions for the slice
            start = max(0, index + start_offset)  # Ensure we don't go out of bounds
            end = index + end_offset  # Ensure we don't go out of bounds

            # Extract epoch
            epoch = raw[0, start:end]  # Keep all columns for the extracted rows
            epochs.append(epoch)
        else:
            break

    evoked = np.mean(np.stack(epochs), axis=0)
    return evoked


def perform_unfold(events_df, erp_data, unfold_model):
    raws = np.ravel(erp_data)

    events_df_jl, bfDict = unfold_model(events_df)

    # Fit Unfold model
    m = Unfold.fit(
        Unfold.UnfoldModel,
        bfDict,
        events_df_jl,
        raws,
        eventcolumn="type",
    )

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

    results_jl = Unfold.coeftable(m)
    results_py = jl_results_to_python(results_jl)

    predicted_raw = Unfold.predict(m).to_numpy()
    predicted_evoked = evoked_from_raw(predicted_raw, events_df, event='go')

    return results_py, predicted_evoked


def basic_model(events_df):
    df = events_df

    bf_baseline = jl.seval("bf_baseline = firbasis(τ = (0, 1.55), sfreq = 64)")
    bf_go = jl.seval("bf_go = firbasis(τ = (-0.2, 0.5), sfreq = 64)")
    bf_stop = jl.seval("bf_stop = firbasis(τ = (-0.2, 0.5), sfreq = 64)")
    bf_response = jl.seval("bf_response_stop = firbasis(τ = (-0.1, 0.6), sfreq = 64)")
    bf_response_nostop = jl.seval("bf_response_nostop = firbasis(τ = (0-0.1, 0.6), sfreq = 64)")

    formula_baseline = jl.seval("f_baseline = @formula 0 ~ baseline")  ######
    formula_go = jl.seval("f_go = @formula 0 ~ 1")
    formula_stop = jl.seval("f_stop = @formula 0 ~ 1 + stop_type + ssd_centered")
    formula_res = jl.seval("f_response_stop = @formula 0 ~ 1 + ssd_centered + sri_centered")
    formula_res_nostop = jl.seval("f_response_nostop = @formula 0 ~ 1")

    bfDict = jl.seval("[ "
                      "\"baseline\" => (f_baseline, bf_baseline), "
                      "\"go\" => (f_go, bf_go), "
                      "\"stop\" => (f_stop, bf_stop),  "
                      "\"response_stop\" => (f_response_stop, bf_response_stop), "
                      "\"response_nostop\" => (f_response_nostop, bf_response_nostop)"
                      "]")

    # Convert the Python columns to Julia arrays
    type_column = jl.seval("Vector{String}")(df['type'].tolist())
    response_type_column = jl.seval("Vector{String}")(df['response_type'].tolist())
    stop_type_column = jl.seval("Vector{String}")(df['stop_type'].tolist())
    ssd_centered_column = jl.seval("Vector{Float64}")(df['ssd'].tolist())
    sri_centered_column = jl.seval("Vector{Float64}")(df['sri'].tolist())
    latency_column = jl.seval("Vector{Int64}")(df['latency'].tolist())
    baseline_column = jl.seval("Vector{Float64}")(df['baseline'].tolist())

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

    return events_df_jl, bfDict


def no_ssd_model(events_df):
    df = events_df

    bf_baseline = jl.seval("bf_baseline = firbasis(τ = (0, 1.55), sfreq = 64)")
    bf_go = jl.seval("bf_go = firbasis(τ = (-0.2, 0.5), sfreq = 64)")
    bf_stop = jl.seval("bf_stop = firbasis(τ = (-0.2, 0.5), sfreq = 64)")
    bf_response = jl.seval("bf_response_stop = firbasis(τ = (-0.1, 0.6), sfreq = 64)")
    bf_response_nostop = jl.seval("bf_response_nostop = firbasis(τ = (0-0.1, 0.6), sfreq = 64)")

    formula_baseline = jl.seval("f_baseline = @formula 0 ~ baseline")  ######
    formula_go = jl.seval("f_go = @formula 0 ~ 1")
    formula_stop = jl.seval("f_stop = @formula 0 ~ 1 + stop_type ")
    formula_res = jl.seval("f_response_stop = @formula 0 ~ 1  + sri_centered")
    formula_res_nostop = jl.seval("f_response_nostop = @formula 0 ~ 1")

    bfDict = jl.seval("[ "
                      "\"baseline\" => (f_baseline, bf_baseline), "
                      "\"go\" => (f_go, bf_go), "
                      "\"stop\" => (f_stop, bf_stop),  "
                      "\"response_stop\" => (f_response_stop, bf_response_stop), "
                      "\"response_nostop\" => (f_response_nostop, bf_response_nostop)"
                      "]")

    # Convert the Python columns to Julia arrays
    type_column = jl.seval("Vector{String}")(df['type'].tolist())
    response_type_column = jl.seval("Vector{String}")(df['response_type'].tolist())
    stop_type_column = jl.seval("Vector{String}")(events_df['stop_type'].tolist())
    ssd_centered_column = jl.seval("Vector{Float64}")(df['ssd'].tolist())
    sri_centered_column = jl.seval("Vector{Float64}")(df['sri'].tolist())
    latency_column = jl.seval("Vector{Int64}")(df['latency'].tolist())
    baseline_column = jl.seval("Vector{Float64}")(df['baseline'].tolist())

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

    return events_df_jl, bfDict


def no_sri_model(events_df):
    df = events_df

    bf_baseline = jl.seval("bf_baseline = firbasis(τ = (0, 1.55), sfreq = 64)")
    bf_go = jl.seval("bf_go = firbasis(τ = (-0.2, 0.5), sfreq = 64)")
    bf_stop = jl.seval("bf_stop = firbasis(τ = (-0.2, 0.5), sfreq = 64)")
    bf_response = jl.seval("bf_response_stop = firbasis(τ = (-0.1, 0.6), sfreq = 64)")
    bf_response_nostop = jl.seval("bf_response_nostop = firbasis(τ = (0-0.1, 0.6), sfreq = 64)")

    formula_baseline = jl.seval("f_baseline = @formula 0 ~ baseline")  ######
    formula_go = jl.seval("f_go = @formula 0 ~ 1")
    formula_stop = jl.seval("f_stop = @formula 0 ~ 1 + stop_type + ssd_centered")
    formula_res = jl.seval("f_response_stop = @formula 0 ~ 1 + ssd_centered")
    formula_res_nostop = jl.seval("f_response_nostop = @formula 0 ~ 1")

    bfDict = jl.seval("[ "
                      "\"baseline\" => (f_baseline, bf_baseline), "
                      "\"go\" => (f_go, bf_go), "
                      "\"stop\" => (f_stop, bf_stop),  "
                      "\"response_stop\" => (f_response_stop, bf_response_stop), "
                      "\"response_nostop\" => (f_response_nostop, bf_response_nostop)"
                      "]")

    # Convert the Python columns to Julia arrays
    type_column = jl.seval("Vector{String}")(df['type'].tolist())
    response_type_column = jl.seval("Vector{String}")(df['response_type'].tolist())
    stop_type_column = jl.seval("Vector{String}")(events_df['stop_type'].tolist())
    ssd_centered_column = jl.seval("Vector{Float64}")(df['ssd'].tolist())
    sri_centered_column = jl.seval("Vector{Float64}")(df['sri'].tolist())
    latency_column = jl.seval("Vector{Int64}")(df['latency'].tolist())
    baseline_column = jl.seval("Vector{Float64}")(df['baseline'].tolist())

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

    return events_df_jl, bfDict


def no_ssd_sri_model(events_df):
    df = events_df

    bf_baseline = jl.seval("bf_baseline = firbasis(τ = (0, 1.55), sfreq = 64)")
    bf_go = jl.seval("bf_go = firbasis(τ = (-0.2, 0.5), sfreq = 64)")
    bf_stop = jl.seval("bf_stop = firbasis(τ = (-0.2, 0.5), sfreq = 64)")
    bf_response = jl.seval("bf_response_stop = firbasis(τ = (-0.1, 0.6), sfreq = 64)")
    bf_response_nostop = jl.seval("bf_response_nostop = firbasis(τ = (0-0.1, 0.6), sfreq = 64)")

    formula_baseline = jl.seval("f_baseline = @formula 0 ~ baseline")  ######
    formula_go = jl.seval("f_go = @formula 0 ~ 1")
    formula_stop = jl.seval("f_stop = @formula 0 ~ 1 + stop_type")
    formula_res = jl.seval("f_response_stop = @formula 0 ~ 1")
    formula_res_nostop = jl.seval("f_response_nostop = @formula 0 ~ 1")

    bfDict = jl.seval("[ "
                      "\"baseline\" => (f_baseline, bf_baseline), "
                      "\"go\" => (f_go, bf_go), "
                      "\"stop\" => (f_stop, bf_stop),  "
                      "\"response_stop\" => (f_response_stop, bf_response_stop), "
                      "\"response_nostop\" => (f_response_nostop, bf_response_nostop)"
                      "]")

    # Convert the Python columns to Julia arrays
    type_column = jl.seval("Vector{String}")(df['type'].tolist())
    response_type_column = jl.seval("Vector{String}")(df['response_type'].tolist())
    stop_type_column = jl.seval("Vector{String}")(events_df['stop_type'].tolist())
    ssd_centered_column = jl.seval("Vector{Float64}")(df['ssd'].tolist())
    sri_centered_column = jl.seval("Vector{Float64}")(df['sri'].tolist())
    latency_column = jl.seval("Vector{Int64}")(df['latency'].tolist())
    baseline_column = jl.seval("Vector{Float64}")(df['baseline'].tolist())

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

    return events_df_jl, bfDict
