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


class UnfoldModelPy:
    def __init__(self, u_model):
        self.u_model = u_model
        self.num_parameters = None
        self.coeftable = None

    def fit(self, signal, events_df):
        signal = np.ravel(signal)
        events_df_jl = get_events_jl(events_df)

        # Fit Unfold model
        m = Unfold.fit(
            Unfold.UnfoldModel,
            self.u_model,
            events_df_jl,
            signal,
            eventcolumn="type",
        )

        self.num_parameters = Unfold.modelmatrix(m).shape[-1]

        results_jl = Unfold.coeftable(m)
        self.coeftable = jl_results_to_python(results_jl)

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


def prepare_data_CV(raw, events_df, start_tp, stop_tp):
    df = events_df.sort_values(by=['latency', 'type'], ascending=[True, False], inplace=False).reset_index()

    # Split events_df into trials around go
    go_indices = df.index[df['type'] == 'go'].tolist()
    go_indices.append(len(df))

    # Split the DataFrame
    dfs = []
    for i in range(len(go_indices) - 1):
        start_idx = go_indices[i]
        end_idx = go_indices[i + 1]
        temp_df = df.iloc[start_idx:end_idx]
        start = temp_df[temp_df['type'] == 'go']['latency'].to_numpy()[0]
        temp_df['latency'] = temp_df['latency'].apply(lambda x: x - start + 13)
        dfs.append(temp_df)

    # split raw signal
    epoched_raw = epochs_from_raw(raw, df, start_tp=-13, stop_tp=101)

    data_df = pd.DataFrame({
        'trial_number': np.arange(0, len(go_indices) - 1),
        'event_df': dfs,
        'eeg_signal': list(epoched_raw)
    })

    return data_df


def create_events_cv(events_df, indices, length):
    events_split = []
    for i, data_index in enumerate(indices):
        trial_events_df = events_df.iloc[i].copy()
        offset = i * length
        trial_events_df['latency'] = trial_events_df['latency'].apply(lambda x: x + offset)
        events_split.append(trial_events_df)

    events_split_df = pd.concat(events_split, ignore_index=True)

    return events_split_df


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

    # Loop through each index to extract the slices
    for index in events_indexes:
        # Calculate the start and end positions for the slice
        start = index + start_offset
        end = index + end_offset

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
    return epochs


def perform_unfold(events_df, erp_data, unfold_model):
    raws = np.ravel(erp_data)

    events_df_jl = get_events_jl(events_df)
    bf_dict = unfold_model

    # Fit Unfold model
    m = Unfold.fit(
        Unfold.UnfoldModel,
        bf_dict,
        events_df_jl,
        raws,
        eventcolumn="type",
    )

    num_parameters = Unfold.modelmatrix(m).shape[-1]

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
    predicted_evoked_df = pd.DataFrame({
        'channel': 1,
        'coefname': 'predicted',
        'estimate': predicted_evoked,
        'eventname': 'evoked',
        'group': None,
        'stderror': None,
        'time': np.linspace(-0.2, 1.58, predicted_evoked.shape[-1])
    })
    original_evoked = evoked_from_raw(erp_data, events_df, event='go')
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

    results_df = pd.concat([results_py, predicted_evoked_df, original_evoked_df, intercept_evoked_df], axis=0,
                           ignore_index=True)

    return results_df, num_parameters


def perform_unfold_cv(events_df_train, erp_data_train, events_df_test, erp_data_test, unfold_model):
    raws_train = np.ravel(erp_data_train)
    raws_test = np.ravel(erp_data_test)

    events_df_jl_train = get_events_jl(events_df_train)
    events_df_jl_test = get_events_jl(events_df_test)

    bf_dict = unfold_model

    # Fit Unfold model
    m = Unfold.fit(
        Unfold.UnfoldModel,
        bf_dict,
        events_df_jl_train,
        raws_train,
        eventcolumn="type",
    )

    num_parameters = Unfold.modelmatrix(m).shape[-1]

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

    predicted_raw_train = Unfold.predict(m).to_numpy()
    # predicted_raw_test = Unfold.predict(m, evts=events_df_jl_test, epoch_to=["go"], eventcolumn="type").to_numpy()
    # print(f"Pred raw shape: {predicted_raw_test.shape}")

    #####
    m_test = Unfold.fit(
        Unfold.UnfoldModel,
        bf_dict,
        events_df_jl_test,
        raws_test,
        eventcolumn="type",
    )
    model_matrix_test = Unfold.modelmatrix(m_test)
    betas_train = results_py.copy()['estimate'].to_numpy()

    predicted_raw_test_manual = np.dot(model_matrix_test, betas_train).reshape(1, -1)
    ######

    predicted_evoked_train = evoked_from_raw(predicted_raw_train, events_df_train, event='go')
    predicted_evoked_df_train = pd.DataFrame({
        'channel': 1,
        'coefname': 'predicted',
        'estimate': predicted_evoked_train,
        'eventname': 'train',
        'group': None,
        'stderror': None,
        'time': np.linspace(-0.2, 1.58, predicted_evoked_train.shape[-1])
    })
    original_evoked_train = evoked_from_raw(erp_data_train, events_df_train, event='go')
    original_evoked_df_train = pd.DataFrame({
        'channel': 1,
        'coefname': 'original',
        'estimate': original_evoked_train,
        'eventname': 'train',
        'group': None,
        'stderror': None,
        'time': np.linspace(-0.2, 1.58, original_evoked_train.shape[-1])
    })
    intercept_evoked_df_train = pd.DataFrame({
        'channel': 1,
        'coefname': '(Intercept)',
        'estimate': np.zeros(original_evoked_train.shape),
        'eventname': 'train',
        'group': None,
        'stderror': None,
        'time': np.linspace(-0.2, 1.58, original_evoked_train.shape[-1])
    })

    ###
    logger.debug(f"ERP data test shape: {erp_data_test.shape}")
    logger.debug(f"ERP data test predicted shape: {predicted_raw_test_manual.shape}")

    predicted_evoked_test = evoked_from_raw(predicted_raw_test_manual, events_df_test, event='go')
    predicted_evoked_df_test = pd.DataFrame({
        'channel': 1,
        'coefname': 'predicted',
        'estimate': predicted_evoked_test,
        'eventname': 'test',
        'group': None,
        'stderror': None,
        'time': np.linspace(-0.2, 1.58, predicted_evoked_test.shape[-1])
    })
    original_evoked_test = evoked_from_raw(erp_data_test, events_df_test, event='go')
    original_evoked_df_test = pd.DataFrame({
        'channel': 1,
        'coefname': 'original',
        'estimate': original_evoked_test,
        'eventname': 'test',
        'group': None,
        'stderror': None,
        'time': np.linspace(-0.2, 1.58, original_evoked_test.shape[-1])
    })
    intercept_evoked_df_test = pd.DataFrame({
        'channel': 1,
        'coefname': '(Intercept)',
        'estimate': np.zeros(original_evoked_test.shape),
        'eventname': 'test',
        'group': None,
        'stderror': None,
        'time': np.linspace(-0.2, 1.58, original_evoked_test.shape[-1])
    })

    results_df = pd.concat([results_py, predicted_evoked_df_train, original_evoked_df_train,
                            intercept_evoked_df_train, predicted_evoked_df_test, original_evoked_df_test,
                            intercept_evoked_df_test], axis=0,
                           ignore_index=True)

    return results_df, num_parameters, predicted_raw_train, predicted_raw_test_manual


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
def get_full_stop_type_stop_ssd_model():
    return jl.seval("""
            bf_baseline = firbasis(τ = (0, 1.58), sfreq = 64)
            bf_go = firbasis(τ = (-0.2, 1.58), sfreq = 64)
            bf_stop = firbasis(τ = (-0.2, 0.5), sfreq = 64)
            bf_response_stop = firbasis(τ = (-0.1, 0.6), sfreq = 64)
            bf_response_nostop = firbasis(τ = (0-0.1, 0.6), sfreq = 64)
            f_baseline = @formula 0 ~ baseline
            f_go = @formula 0 ~ 1
            f_stop = @formula 0 ~ 1 + stop_type + ssd_centered
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
def get_response_model():
    return jl.seval("""
            bf_baseline = firbasis(τ = (0, 1.58), sfreq = 64)
            bf_go = firbasis(τ = (-0.2, 1.58), sfreq = 64)
            bf_stop = firbasis(τ = (-0.2, 0.5), sfreq = 64)
            bf_response = firbasis(τ = (-0.1, 0.6), sfreq = 64)
            f_baseline = @formula 0 ~ baseline
            f_go = @formula 0 ~ 1
            f_stop = @formula 0 ~ 1 
            f_response = @formula 0 ~ 1 
            [
                "baseline" => (f_baseline, bf_baseline),    
                "go" => (f_go, bf_go),      
                "stop" => (f_stop, bf_stop), 
                "response" => (f_response, bf_response),
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


# Convert the Python columns to Julia arrays
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

# def no_ssd_model(events_df):
#     df = events_df

#     bf_baseline = jl.seval("bf_baseline = firbasis(τ = (0, 1.55), sfreq = 64)")
#     bf_go = jl.seval("bf_go = firbasis(τ = (-0.2, 0.5), sfreq = 64)")
#     bf_stop = jl.seval("bf_stop = firbasis(τ = (-0.2, 0.5), sfreq = 64)")
#     bf_response = jl.seval("bf_response_stop = firbasis(τ = (-0.1, 0.6), sfreq = 64)")
#     bf_response_nostop = jl.seval("bf_response_nostop = firbasis(τ = (0-0.1, 0.6), sfreq = 64)")

#     formula_baseline = jl.seval("f_baseline = @formula 0 ~ baseline")  ######
#     formula_go = jl.seval("f_go = @formula 0 ~ 1")
#     formula_stop = jl.seval("f_stop = @formula 0 ~ 1 + stop_type ")
#     formula_res = jl.seval("f_response_stop = @formula 0 ~ 1  + sri_centered")
#     formula_res_nostop = jl.seval("f_response_nostop = @formula 0 ~ 1")

#     bfDict = jl.seval("[ "
#                       "\"baseline\" => (f_baseline, bf_baseline), "
#                       "\"go\" => (f_go, bf_go), "
#                       "\"stop\" => (f_stop, bf_stop),  "
#                       "\"response_stop\" => (f_response_stop, bf_response_stop), "
#                       "\"response_nostop\" => (f_response_nostop, bf_response_nostop)"
#                       "]")

#     # Convert the Python columns to Julia arrays
#     type_column = jl.seval("Vector{String}")(df['type'].tolist())
#     response_type_column = jl.seval("Vector{String}")(df['response_type'].tolist())
#     stop_type_column = jl.seval("Vector{String}")(events_df['stop_type'].tolist())
#     ssd_centered_column = jl.seval("Vector{Float64}")(df['ssd'].tolist())
#     sri_centered_column = jl.seval("Vector{Float64}")(df['sri'].tolist())
#     latency_column = jl.seval("Vector{Int64}")(df['latency'].tolist())
#     baseline_column = jl.seval("Vector{Float64}")(df['baseline'].tolist())

#     # Create the Julia DataFrame
#     events_df_jl = jl.DataFrame(
#         type=type_column,
#         latency=latency_column,
#         ssd_centered=ssd_centered_column,
#         sri_centered=sri_centered_column,
#         response_type=response_type_column,
#         stop_type=stop_type_column,
#         baseline=baseline_column
#     )

#     return events_df_jl, bfDict


# def no_sri_model(events_df):
#     df = events_df

#     bf_baseline = jl.seval("bf_baseline = firbasis(τ = (0, 1.55), sfreq = 64)")
#     bf_go = jl.seval("bf_go = firbasis(τ = (-0.2, 0.5), sfreq = 64)")
#     bf_stop = jl.seval("bf_stop = firbasis(τ = (-0.2, 0.5), sfreq = 64)")
#     bf_response = jl.seval("bf_response_stop = firbasis(τ = (-0.1, 0.6), sfreq = 64)")
#     bf_response_nostop = jl.seval("bf_response_nostop = firbasis(τ = (0-0.1, 0.6), sfreq = 64)")

#     formula_baseline = jl.seval("f_baseline = @formula 0 ~ baseline")  ######
#     formula_go = jl.seval("f_go = @formula 0 ~ 1")
#     formula_stop = jl.seval("f_stop = @formula 0 ~ 1 + stop_type + ssd_centered")
#     formula_res = jl.seval("f_response_stop = @formula 0 ~ 1 + ssd_centered")
#     formula_res_nostop = jl.seval("f_response_nostop = @formula 0 ~ 1")

#     bfDict = jl.seval("[ "
#                       "\"baseline\" => (f_baseline, bf_baseline), "
#                       "\"go\" => (f_go, bf_go), "
#                       "\"stop\" => (f_stop, bf_stop),  "
#                       "\"response_stop\" => (f_response_stop, bf_response_stop), "
#                       "\"response_nostop\" => (f_response_nostop, bf_response_nostop)"
#                       "]")

#     # Convert the Python columns to Julia arrays
#     type_column = jl.seval("Vector{String}")(df['type'].tolist())
#     response_type_column = jl.seval("Vector{String}")(df['response_type'].tolist())
#     stop_type_column = jl.seval("Vector{String}")(events_df['stop_type'].tolist())
#     ssd_centered_column = jl.seval("Vector{Float64}")(df['ssd'].tolist())
#     sri_centered_column = jl.seval("Vector{Float64}")(df['sri'].tolist())
#     latency_column = jl.seval("Vector{Int64}")(df['latency'].tolist())
#     baseline_column = jl.seval("Vector{Float64}")(df['baseline'].tolist())

#     # Create the Julia DataFrame
#     events_df_jl = jl.DataFrame(
#         type=type_column,
#         latency=latency_column,
#         ssd_centered=ssd_centered_column,
#         sri_centered=sri_centered_column,
#         response_type=response_type_column,
#         stop_type=stop_type_column,
#         baseline=baseline_column
#     )

#     return events_df_jl, bfDict


# def no_ssd_sri_model(events_df):
#     df = events_df

#     bf_baseline = jl.seval("bf_baseline = firbasis(τ = (0, 1.55), sfreq = 64)")
#     bf_go = jl.seval("bf_go = firbasis(τ = (-0.2, 0.5), sfreq = 64)")
#     bf_stop = jl.seval("bf_stop = firbasis(τ = (-0.2, 0.5), sfreq = 64)")
#     bf_response = jl.seval("bf_response_stop = firbasis(τ = (-0.1, 0.6), sfreq = 64)")
#     bf_response_nostop = jl.seval("bf_response_nostop = firbasis(τ = (0-0.1, 0.6), sfreq = 64)")

#     formula_baseline = jl.seval("f_baseline = @formula 0 ~ baseline")  ######
#     formula_go = jl.seval("f_go = @formula 0 ~ 1")
#     formula_stop = jl.seval("f_stop = @formula 0 ~ 1 + stop_type")
#     formula_res = jl.seval("f_response_stop = @formula 0 ~ 1")
#     formula_res_nostop = jl.seval("f_response_nostop = @formula 0 ~ 1")

#     bfDict = jl.seval("[ "
#                       "\"baseline\" => (f_baseline, bf_baseline), "
#                       "\"go\" => (f_go, bf_go), "
#                       "\"stop\" => (f_stop, bf_stop),  "
#                       "\"response_stop\" => (f_response_stop, bf_response_stop), "
#                       "\"response_nostop\" => (f_response_nostop, bf_response_nostop)"
#                       "]")

#     # Convert the Python columns to Julia arrays
#     type_column = jl.seval("Vector{String}")(df['type'].tolist())
#     response_type_column = jl.seval("Vector{String}")(df['response_type'].tolist())
#     stop_type_column = jl.seval("Vector{String}")(events_df['stop_type'].tolist())
#     ssd_centered_column = jl.seval("Vector{Float64}")(df['ssd'].tolist())
#     sri_centered_column = jl.seval("Vector{Float64}")(df['sri'].tolist())
#     latency_column = jl.seval("Vector{Int64}")(df['latency'].tolist())
#     baseline_column = jl.seval("Vector{Float64}")(df['baseline'].tolist())

#     # Create the Julia DataFrame
#     events_df_jl = jl.DataFrame(
#         type=type_column,
#         latency=latency_column,
#         ssd_centered=ssd_centered_column,
#         sri_centered=sri_centered_column,
#         response_type=response_type_column,
#         stop_type=stop_type_column,
#         baseline=baseline_column
#     )

#     return events_df_jl, bfDict
