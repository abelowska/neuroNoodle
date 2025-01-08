import numpy as np
import pandas as pd
from functools import cache
import logging

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

@cache
def get_full_interaction_model():
    return jl.seval("""
            bf_baseline = firbasis(τ = (0, 1.58), sfreq = 64)
            bf_go = firbasis(τ = (-0.2, 1.58), sfreq = 64)
            bf_stop = firbasis(τ = (-0.2, 0.5), sfreq = 64)
            bf_response_stop = firbasis(τ = (-0.1, 0.6), sfreq = 64)
            bf_response_nostop = firbasis(τ = (0-0.1, 0.6), sfreq = 64)
            f_baseline = @formula 0 ~ baseline
            f_go = @formula 0 ~ 1
            f_stop = @formula 0 ~ 1 + stop_type + ssd_centered
            f_response_stop = @formula 0 ~ 1 + ssd_centered + sri_centered
            f_response_nostop = @formula 0 ~ 1
            [
                "baseline" => (f_baseline, bf_baseline),    
                "go" => (f_go, bf_go),      
                "stop" => (f_stop, bf_stop), 
                "response_stop" => (f_response_stop, bf_response_stop),
                "response_nostop" => (f_response_nostop, bf_response_nostop)
            ]
        """
                    )


@cache
def get_full_stop_type_sri_model():
    return jl.seval("""
            bf_baseline = firbasis(τ = (0, 1.58), sfreq = 64)
            bf_go = firbasis(τ = (-0.2, 1.58), sfreq = 64)
            bf_stop = firbasis(τ = (-0.2, 0.5), sfreq = 64)
            bf_response_stop = firbasis(τ = (-0.1, 0.6), sfreq = 64)
            bf_response_nostop = firbasis(τ = (0-0.1, 0.6), sfreq = 64)
            f_baseline = @formula 0 ~ baseline
            f_go = @formula 0 ~ 1
            f_stop = @formula 0 ~ 1 + stop_type
            f_response_stop = @formula 0 ~ 1 + sri_centered
            f_response_nostop = @formula 0 ~ 1
            [
                "baseline" => (f_baseline, bf_baseline),    
                "go" => (f_go, bf_go),      
                "stop" => (f_stop, bf_stop), 
                "response_stop" => (f_response_stop, bf_response_stop),
                "response_nostop" => (f_response_nostop, bf_response_nostop)
            ]
        """
                    )


@cache
def get_full_stop_type_ssd_model():
    return jl.seval("""
            bf_baseline = firbasis(τ = (0, 1.58), sfreq = 64)
            bf_go = firbasis(τ = (-0.2, 1.58), sfreq = 64)
            bf_stop = firbasis(τ = (-0.2, 0.5), sfreq = 64)
            bf_response_stop = firbasis(τ = (-0.1, 0.6), sfreq = 64)
            bf_response_nostop = firbasis(τ = (0-0.1, 0.6), sfreq = 64)
            f_baseline = @formula 0 ~ baseline
            f_go = @formula 0 ~ 1
            f_stop = @formula 0 ~ 1 + stop_type + ssd_centered
            f_response_stop = @formula 0 ~ 1 + ssd_centered
            f_response_nostop = @formula 0 ~ 1
            [
                "baseline" => (f_baseline, bf_baseline),    
                "go" => (f_go, bf_go),      
                "stop" => (f_stop, bf_stop), 
                "response_stop" => (f_response_stop, bf_response_stop),
                "response_nostop" => (f_response_nostop, bf_response_nostop)
            ]
        """
                    )


@cache
def get_full_stop_type_model():
    return jl.seval("""
            bf_baseline = firbasis(τ = (0, 1.58), sfreq = 64)
            bf_go = firbasis(τ = (-0.2, 1.58), sfreq = 64)
            bf_stop = firbasis(τ = (-0.2, 0.5), sfreq = 64)
            bf_response_stop = firbasis(τ = (-0.1, 0.6), sfreq = 64)
            bf_response_nostop = firbasis(τ = (0-0.1, 0.6), sfreq = 64)
            f_baseline = @formula 0 ~ baseline
            f_go = @formula 0 ~ 1
            f_stop = @formula 0 ~ 1 + stop_type 
            f_response_stop = @formula 0 ~ 1 
            f_response_nostop = @formula 0 ~ 1
            [
                "baseline" => (f_baseline, bf_baseline),    
                "go" => (f_go, bf_go),      
                "stop" => (f_stop, bf_stop), 
                "response_stop" => (f_response_stop, bf_response_stop),
                "response_nostop" => (f_response_nostop, bf_response_nostop)
            ]
        """
                    )


@cache
def get_full_model():
    return jl.seval("""
            bf_baseline = firbasis(τ = (0, 1.58), sfreq = 64)
            bf_go = firbasis(τ = (-0.2, 1.58), sfreq = 64)
            bf_stop = firbasis(τ = (-0.2, 0.5), sfreq = 64)
            bf_response_stop = firbasis(τ = (-0.1, 0.6), sfreq = 64)
            bf_response_nostop = firbasis(τ = (0-0.1, 0.6), sfreq = 64)
            f_baseline = @formula 0 ~ baseline
            f_go = @formula 0 ~ 1
            f_stop = @formula 0 ~ 1 
            f_response_stop = @formula 0 ~ 1 
            f_response_nostop = @formula 0 ~ 1
            [
                "baseline" => (f_baseline, bf_baseline),    
                "go" => (f_go, bf_go),      
                "stop" => (f_stop, bf_stop), 
                "response_stop" => (f_response_stop, bf_response_stop),
                "response_nostop" => (f_response_nostop, bf_response_nostop)
            ]
        """
                    )


@cache
def get_stop_model():
    return jl.seval("""
            bf_baseline = firbasis(τ = (0, 1.58), sfreq = 64)
            bf_go = firbasis(τ = (-0.2, 1.58), sfreq = 64)
            bf_stop = firbasis(τ = (-0.2, 0.5), sfreq = 64)
           
            f_baseline = @formula 0 ~ baseline
            f_go = @formula 0 ~ 1
            f_stop = @formula 0 ~ 1 
            
            [
                "baseline" => (f_baseline, bf_baseline),    
                "go" => (f_go, bf_go),      
                "stop" => (f_stop, bf_stop), 
            ]
        """
                    )


@cache
def get_go_model():
    return jl.seval("""
            bf_baseline = firbasis(τ = (0, 1.58), sfreq = 64)
            bf_go = firbasis(τ = (-0.2, 1.58), sfreq = 64)
           
            f_baseline = @formula 0 ~ baseline
            f_go = @formula 0 ~ 1
          
            [
                "baseline" => (f_baseline, bf_baseline),    
                "go" => (f_go, bf_go),      
            ]
        """
                    )


@cache
def get_baseline_model():
    return jl.seval("""
            bf_baseline = firbasis(τ = (0, 1.58), sfreq = 64)
            f_baseline = @formula 0 ~ baseline
            [
                "baseline" => (f_baseline, bf_baseline),    
            ]
        """
                    )