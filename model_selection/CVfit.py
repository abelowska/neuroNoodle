from sklearn.model_selection import ShuffleSplit
import logging
import sys
import pandas as pd
import numpy as np
import pickle
from unfold.simple_ridgeCV_model import UnfoldSimpleRidgeCVModelPy

sys.path.insert(1, '../')

from unfold import unfold_model, unfold_utils

logger = logging.getLogger(__name__)


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
    epoched_raw = unfold_model.epochs_from_raw(raw, df, start_tp=-13, stop_tp=101)

    assert len(np.arange(0, len(go_indices) - 1)) == len(dfs) == len(list(epoched_raw)), ('Trial numbers not equals '
                                                                                          'number of events and '
                                                                                          'signal epochs')

    data_df = pd.DataFrame({
        'trial_number': np.arange(0, len(go_indices) - 1),
        'event_df': dfs,
        'eeg_signal': list(epoched_raw)
    })

    return data_df


def create_events_cv(events_df, indices, length, return_trial=True):
    events_split = []
    events_split_trial_level = []
    for i, data_index in enumerate(indices):
        # logger.debug(f"EVENTS CV:\n fold num: {i}, data index: {data_index}")
        trial_events_df = events_df.iloc[data_index].copy()
        offset = i * length
        trial_events_df['latency'] = trial_events_df['latency'].apply(lambda x: x + offset)
        events_split.append(trial_events_df)
        events_split_trial_level.append(events_df.iloc[data_index].copy())

    events_split_df = pd.concat(events_split, ignore_index=True)

    if return_trial:
        return events_split_df, events_split_trial_level
    else:
        return events_split_df


class UnfoldModelPyCV:
    def __init__(self, estimator):
        self.estimator = estimator
        self.channels = None
        self.cv_results = {}
        self.channel_results = {}
        self.results = None
        self.num_parameters = None

    def fit_unfold_model(
            self,
            raw,
            events_df,
            cv=ShuffleSplit(n_splits=20, random_state=42, test_size=0.50),
            channels='all',
            as_epochs=True,
            refit=False,
    ):
        """

        Parameters
        ----------
        raw : mne.Raw
        events_df : pd.DataFrame
        cv : splitter
        channels : list
        as_epochs :
        refit :

        Returns
        -------

        """
        # perform unfolding separately for each channel
        if channels == 'all':
            self.channels = raw.info['ch_names']
        else:
            self.channels = channels

        for channel in self.channels:
            channel_events_df = events_df[channel]
            # clear bad segments
            channel_events_df = channel_events_df[channel_events_df['bad'] == False]
            # get EEG data
            eeg_data = raw.get_data(picks=channel)
            # cast EEG data from V to uV
            eeg_data = eeg_data * 1000000

            # fit channel data
            self.fit_channel(channel, eeg_data, channel_events_df, cv=cv, as_epochs=as_epochs,
                             refit=refit)

        if self.channel_results:
            logger.debug(f"In saving channel results into ResultsContainer")
            results_obj = unfold_utils.ResultsContainer()
            results_obj.num_parameters = self.num_parameters
            results_obj.process_results(self.channel_results)
            self.results = results_obj

    def fit_channel(
            self,
            channel_name,
            eeg_data,
            events_df,
            cv=ShuffleSplit(n_splits=20, random_state=42, test_size=0.50),
            as_epochs=True,
            refit=False,
    ):
        if cv is not None:
            cv_predicted_train = []
            cv_original_train = []
            cv_predicted_test = []
            cv_original_test = []
            split_num = []
            alphas = []
            cv_events_train = []
            cv_events_test = []

            # prepare data for splitting - divide events-df and raw signal into trials
            data_df = prepare_data_CV(eeg_data, events_df, start_tp=-13, stop_tp=101)

            # generate splits and perform unfold
            for i, (train_index, test_index) in enumerate(cv.split(data_df)):
                split_num.append(i)
                logger.debug(f"Estimating fold: {i}")
                logger.info(f"Train indices: {train_index}")
                logger.info(f"Test indices: {test_index}")

                # create data for split
                epochs_train = np.vstack(data_df['eeg_signal'].to_numpy()[train_index])
                length = epochs_train[0].shape[-1]
                logger.debug(f"Length of the epoch: {length}")

                continuous_train = np.vstack(epochs_train).flatten().reshape(1, -1)
                logger.debug(f"Continuous train shape: {continuous_train.shape}")
                channel_events_train_df, channel_events_trials_train = create_events_cv(data_df['event_df'], train_index, length)

                epochs_test = np.vstack(data_df['eeg_signal'].to_numpy()[test_index])
                continuous_test = np.vstack(epochs_test).flatten().reshape(1, -1)
                logger.debug(f"Continuous test shape: {continuous_test.shape}")
                channel_events_test_df, channel_events_trials_test = create_events_cv(data_df['event_df'], test_index, length)

                # perform unfold with train/test
                self.estimator.fit(signal=continuous_train, events_df=channel_events_train_df)

                # predict on the train and test data
                predicted_raw_train = self.estimator.predict(signal=continuous_train, events_df=channel_events_train_df)
                predicted_raw_test = self.estimator.predict(signal=continuous_test, events_df=channel_events_test_df)

                # Add info on chosen alpha in the split
                if hasattr(self.estimator, 'linear_estimator'):
                    alphas.append(self.estimator.linear_estimator.best_alphas_)
                else:
                    alphas.append(None)

                # add events from the split
                cv_events_train.append(channel_events_trials_train)
                cv_events_test.append(channel_events_trials_test)

                if as_epochs:
                    if predicted_raw_train.shape == epochs_train.shape:
                        cv_predicted_train.append(
                            np.array(predicted_raw_train))
                        cv_predicted_test.append(
                            np.array(predicted_raw_test))
                    else:
                        cv_predicted_train.append(
                            np.array(unfold_model.epochs_from_raw(predicted_raw_train, channel_events_train_df)))
                        cv_predicted_test.append(
                            np.array(unfold_model.epochs_from_raw(predicted_raw_test, channel_events_test_df)))
                    cv_original_train.append(np.array(epochs_train))
                    cv_original_test.append(np.array(epochs_test))
                else:
                    cv_predicted_train.append(np.array(predicted_raw_train))
                    cv_predicted_test.append(np.array(predicted_raw_test))
                    cv_original_train.append(np.array(continuous_train))
                    cv_original_test.append(np.array(continuous_test))

            # TODO: change it to assert
            logger.debug(f"split len: {len(split_num)}\nalphas len: {len(alphas)}\ntrain events len: {len(cv_events_train)}\ntest events len:{len(cv_events_test)}")

            # create results df
            cv_results_df = pd.DataFrame({
                'split_num': split_num,
                'alphas': alphas,
                'train_original': cv_original_train,
                'train_predicted': cv_predicted_train,
                'test_original': cv_original_test,
                'test_predicted': cv_predicted_test,
                'train_events': cv_events_train,
                'test_events': cv_events_test
            })

            self.cv_results[channel_name] = cv_results_df

            if refit:
                self.estimator.fit(signal=eeg_data, events_df=events_df)
                self.num_parameters = self.estimator.num_parameters

                results_channel_df = self.estimator.get_results(predicted=True, signal=eeg_data, events_df=events_df)
                self.channel_results[channel_name] = results_channel_df

        else:
            self.estimator.fit(signal=eeg_data, events_df=events_df)
            self.num_parameters = self.estimator.num_parameters

            results_channel_df = self.estimator.get_results(predicted=True, signal=eeg_data, events_df=events_df)
            self.channel_results[channel_name] = results_channel_df

    def save_model_results(self, path):
        self.results.store_results(path)

    def save_cv_results(self, path):
        with open(f'{path}_cv.pickle', 'wb') as handle:
            pickle.dump(self.cv_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
