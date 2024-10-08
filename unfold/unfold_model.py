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


def perform_unfold(events_df, erp_data):
    df = events_df

    bf_baseline = jl.seval("bf_baseline = firbasis(τ = (0, 1.55), sfreq = 64)")
    bf_go = jl.seval("bf_go = firbasis(τ = (-0.2, 0.5), sfreq = 64)")
    bf_stop = jl.seval("bf_stop = firbasis(τ = (-0.2, 0.5), sfreq = 64)")
    bf_response = jl.seval("bf_response_stop = firbasis(τ = (-0.1, 0.6), sfreq = 64)")
    bf_response_nostop = jl.seval("bf_response_nostop = firbasis(τ = (0-0.1, 0.6), sfreq = 64)")

    formula_baseline = jl.seval("f_baseline = @formula 0 ~ 1")
    formula_go = jl.seval("f_go = @formula 0 ~ 1")
    formula_stop = jl.seval("f_stop = @formula 0 ~ 1 + stop_type + ssd_centered")
    formula_res = jl.seval("f_response_stop = @formula 0 ~ 1 + sri_centered + ssd_centered")
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

    # Create the Julia DataFrame
    events_df_jl = jl.DataFrame(
        type=type_column,
        latency=latency_column,
        ssd_centered=ssd_centered_column,
        sri_centered=sri_centered_column,
        response_type=response_type_column,
        stop_type=stop_type_column,
    )

    raws = np.ravel(erp_data)

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
