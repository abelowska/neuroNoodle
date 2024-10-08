import seaborn as sns
import matplotlib.pyplot as plt
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


def perform_unfold(events_df, erp_trials):
    df = events_df

    bf_go = jl.seval("bf_go = firbasis(τ = (-0.2, 0.5), sfreq = 64)")
    bf_stop = jl.seval("bf_stop = firbasis(τ = (-0.2, 0.5), sfreq = 64)")
    bf_response = jl.seval("bf_response = firbasis(τ = (-0.1, 0.6), sfreq = 64)")
    bf_response_nostop = jl.seval("bf_response_nostop = firbasis(τ = (0-0.1, 0.6), sfreq = 64)")

    formula_go = jl.seval("f_go = @formula 0 ~ 1")
    formula_stop = jl.seval("f_stop = @formula 0 ~ 1 + stop_type + ssd_centered")
    formula_res = jl.seval("f_response = @formula 0 ~ 1 + sri_centered + ssd_centered")
    formula_res_nostop = jl.seval("f_response_nostop = @formula 0 ~ 1")

    bfDict = jl.seval(
        "[ \"go\" => (f_go, bf_go), \"stop\" => (f_stop, bf_stop),  \"response\" => (f_response, bf_response), "
        "\"response_nostop\" => (f_response_nostop, bf_response_nostop)]")

    # Convert the Python columns to Julia arrays
    type_column = jl.seval("Vector{String}")(df['event'].tolist())
    response_type_column = jl.seval("Vector{String}")(df['response_type'].tolist())
    stop_type_column = jl.seval("Vector{String}")(events_df['stop_type'].tolist())
    ssd_centered_column = jl.seval("Vector{Float64}")(df['SSD'].tolist())
    sri_centered_column = jl.seval("Vector{Float64}")(df['SRI'].tolist())
    latency_column = jl.seval("Vector{Int64}")(df['latency'].tolist())

    # Create the Julia DataFrame
    events_df_jl = jl.DataFrame(
        type=type_column,
        latency=latency_column,
        ssd_centered=ssd_centered_column,
        sri_centered=sri_centered_column,
        response_type=response_type_column,
        stop_type=stop_type_column,
    )

    raws = np.ravel(erp_trials)

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

    return results_py


def perform_unfold2(events_df, erp_trials):
    df = events_df

    bf_go = jl.seval("bf_go = firbasis(τ = (-0.2, 0.5), sfreq = 64)")
    bf_stop = jl.seval("bf_stop = firbasis(τ = (-0.2, 0.5), sfreq = 64)")
    bf_response = jl.seval("bf_response = firbasis(τ = (-0.1, 0.6), sfreq = 64)")
    bf_response_nostop = jl.seval("bf_response_nostop = firbasis(τ = (0-0.1, 0.6), sfreq = 64)")

    formula_go = jl.seval("f_go = @formula 0 ~ 1")
    formula_stop = jl.seval("f_stop = @formula 0 ~ 1 + ssd_centered")
    formula_res = jl.seval("f_response = @formula 0 ~ 1 + sri_centered + ssd_centered")
    formula_res_nostop = jl.seval("f_response_nostop = @formula 0 ~ 1")

    bfDict = jl.seval(
        "[ \"go\" => (f_go, bf_go), \"stop\" => (f_stop, bf_stop),  \"response\" => (f_response, bf_response), "
        "\"response_nostop\" => (f_response_nostop, bf_response_nostop)]")

    # Convert the Python columns to Julia arrays
    type_column = jl.seval("Vector{String}")(df['event'].tolist())
    response_type_column = jl.seval("Vector{String}")(df['response_type'].tolist())
    stop_type_column = jl.seval("Vector{String}")(events_df['stop_type'].tolist())
    ssd_centered_column = jl.seval("Vector{Float64}")(df['SSD'].tolist())
    sri_centered_column = jl.seval("Vector{Float64}")(df['SRI'].tolist())
    latency_column = jl.seval("Vector{Int64}")(df['latency'].tolist())

    # Create the Julia DataFrame
    events_df_jl = jl.DataFrame(
        type=type_column,
        latency=latency_column,
        ssd_centered=ssd_centered_column,
        sri_centered=sri_centered_column,
        response_type=response_type_column,
        stop_type=stop_type_column,
    )

    raws = np.ravel(erp_trials)

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

    return results_py

def plot_unfold_results_sst(results):
    # Extract the coefficients for one channel
    results_channel = results[results.channel == 1]

    results_go = results_channel[results_channel.eventname == 'go']
    results_stop = results_channel[results_channel.eventname == 'stop']
    results_response = results_channel[results_channel.eventname == 'response']
    results_response_nostop = results_channel[results_channel.eventname == 'response_nostop']

    sns.set_style("whitegrid")

    # Set global font size for various elements
    plt.rcParams.update({
        'font.size': 25,
        'axes.titlesize': 30,
        'axes.labelsize': 30,
        'xtick.labelsize': 30,
        'ytick.labelsize': 30,
        'legend.fontsize': 25,
        'figure.titlesize': 25,
    })
    linewidth = 5
    # Plot the coefficient estimates over time
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(30, 20), sharey=True)

    ax1 = sns.lineplot(
        x=results_go.time,
        y=results_go.estimate,
        hue=results_go.coefname,
        ax=ax1,
        linewidth=linewidth
    )
    ax1.set(xlabel='Time [s]', ylabel='Coefficient estimate', title='Go')

    ax2 = sns.lineplot(
        x=results_stop.time,
        y=results_stop.estimate,
        hue=results_stop.coefname,
        ax=ax2,
        linewidth=linewidth
    )
    ax2.set(xlabel='Time [s]', ylabel='Coefficient estimate', title='Stop')

    ax3 = sns.lineplot(
        x=results_response.time,
        y=results_response.estimate,
        hue=results_response.coefname,
        ax=ax3,
        linewidth=linewidth
    )
    ax3.set(xlabel='Time [s]', ylabel='Coefficient estimate', title='Response')

    ax4 = sns.lineplot(
        x=results_response_nostop.time,
        y=results_response_nostop.estimate,
        hue=results_response_nostop.coefname,
        ax=ax4,
        linewidth=linewidth
    )
    ax4.set(xlabel='Time [s]', ylabel='Coefficient estimate', title='Response nostop')

    plt.tight_layout()
    plt.show()
