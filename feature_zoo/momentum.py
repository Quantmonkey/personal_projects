import os.path
ABS_PATH_TO_CACHE='C:/Users/Oscar/git/transparent_offchain/main/models/features/cache/'

import pickle
import numpy as np
import pandas as pd
from ta.momentum import *
from sklearn.preprocessing import StandardScaler

from ..utils import *

class WrappedAwesomeOscillatorIndicator:
    def __init__(self,
                 high_atom_name,
                 low_atom_name,
                 clustering_atom_name,
                 window1,
                 window2,
                 rets_form_list,
                 reference,
                 reference_window,
                 windows_buffer_mult):
        self.high_atom_name = high_atom_name
        self.low_atom_name = low_atom_name
        self.clustering_atom_name = clustering_atom_name
        self.window1 = window1
        self.window2 = window2
        self.rets_form_list = rets_form_list
        self.reference_window = reference_window
        self.reference = reference
        self.standard_scaler = StandardScaler()
        self.windows_buffer_mult = windows_buffer_mult

    def transform(self,
                  atoms,
                  start_feature_data_date,
                  prediction_date,
                  realization_date,
                  reference_id,
                  active_cross_section_series,
                  universe_name):

        params_list = [prediction_date, 
                       start_feature_data_date,
                       self.high_atom_name,
                       self.low_atom_name,
                       self.window1,
                       self.window2,
                       self.rets_form_list,
                       self.reference_window]

        path_to_file = get_path_to_file(
            params_list=params_list, 
            ABS_PATH_TO_CACHE=ABS_PATH_TO_CACHE+universe_name+'/', 
            unique_name='WrappedAwesomeOscillatorIndicator')

        indicator = get_pickled_indicator(path_to_file)
        if indicator.empty:
            atom_name_list = [self.high_atom_name,
                              self.low_atom_name]
            windows_list = [self.window1,
                            self.window2,
                            self.reference_window]
                            
            (high_atom, low_atom) = get_processed_atoms(
                atoms=atoms, 
                atom_name_list=atom_name_list, 
                start_feature_data_date=start_feature_data_date, 
                prediction_date=prediction_date, 
                windows_list=windows_list, 
                rets_form_list=self.rets_form_list,
                windows_buffer_mult=self.windows_buffer_mult,
                active_cross_section=active_cross_section_series[
                    prediction_date
                ]
            )

            indicator = high_atom.apply(
                lambda x: AwesomeOscillatorIndicator(
                    high=x,
                    low=low_atom[x.name],
                    window1=self.window1,
                    window2=self.window2,
                ).awesome_oscillator()
            ).iloc[-self.reference_window:].replace({np.inf: np.nan, -np.inf: np.nan}).dropna(how='all', axis=1)
            pickle.dump(indicator, open(path_to_file, 'wb'))

        if self.reference == 'cross':
            (sec_to_cluster, 
            cluster_to_sec) = get_clusters(
                prediction_date=prediction_date, 
                clustering_atom=atoms[self.clustering_atom_name],
                active_cross_section=active_cross_section_series[
                    prediction_date
                ],
                universe_name=universe_name,
            )
        else:
            sec_to_cluster = None
            cluster_to_sec = None

        scaled_indicator = get_scaled_indicator(
            indicator=indicator, 
            scaler=self.standard_scaler, 
            reference=self.reference, 
            reference_window=self.reference_window, 
            reference_id=reference_id,
            sec_to_cluster=sec_to_cluster,
            cluster_to_sec=cluster_to_sec,
        )

        return scaled_indicator

class WrappedKAMAIndicator:
    def __init__(self,
                 close_atom_name,
                 clustering_atom_name,
                 window,
                 pow1,
                 pow2,
                 rets_form_list,
                 reference,
                 reference_window,
                 windows_buffer_mult):
        self.close_atom_name = close_atom_name
        self.clustering_atom_name = clustering_atom_name
        self.window = window
        self.pow1 = pow1
        self.pow2 = pow2
        self.rets_form_list = rets_form_list
        self.reference_window = reference_window
        self.reference = reference
        self.standard_scaler = StandardScaler()
        self.windows_buffer_mult = windows_buffer_mult

    def transform(self,
                  atoms,
                  start_feature_data_date,
                  prediction_date,
                  realization_date,
                  reference_id,
                  active_cross_section_series,
                  universe_name):

        params_list = [prediction_date, 
                       start_feature_data_date,
                       self.close_atom_name,
                       self.window,
                       self.pow1,
                       self.pow2,
                       self.rets_form_list,
                       self.reference_window]

        path_to_file = get_path_to_file(
            params_list=params_list, 
            ABS_PATH_TO_CACHE=ABS_PATH_TO_CACHE+universe_name+'/', 
            unique_name='WrappedKAMAIndicator')
            
        indicator = get_pickled_indicator(path_to_file)
        if indicator.empty:
            atom_name_list = [self.close_atom_name,]
            windows_list = [self.window,
                            self.pow1,
                            self.pow2,
                            self.reference_window]
                            
            (close_atom,) = get_processed_atoms(
                atoms=atoms, 
                atom_name_list=atom_name_list, 
                start_feature_data_date=start_feature_data_date, 
                prediction_date=prediction_date, 
                windows_list=windows_list, 
                rets_form_list=self.rets_form_list,
                windows_buffer_mult=self.windows_buffer_mult,
                active_cross_section=active_cross_section_series[
                    prediction_date
                ]
            )
            indicator = close_atom.apply(
                lambda x: KAMAIndicator(
                    close=x,
                    window=self.window,
                    pow1=self.pow1,
                    pow2=self.pow2,
                ).kama()
            ).iloc[-self.reference_window:].replace({np.inf: np.nan, -np.inf: np.nan}).dropna(how='all', axis=1)
            indicator = (
                close_atom.loc[indicator.index]
                -indicator
            ).dropna(how='all', axis=1)
            pickle.dump(indicator, open(path_to_file, 'wb'))

        if self.reference == 'cross':
            (sec_to_cluster, 
            cluster_to_sec) = get_clusters(
                prediction_date=prediction_date, 
                clustering_atom=atoms[self.clustering_atom_name],
                active_cross_section=active_cross_section_series[
                    prediction_date
                ],
                universe_name=universe_name,
            )
        else:
            sec_to_cluster = None
            cluster_to_sec = None

        scaled_indicator = get_scaled_indicator(
            indicator=indicator, 
            scaler=self.standard_scaler, 
            reference=self.reference, 
            reference_window=self.reference_window, 
            reference_id=reference_id,
            sec_to_cluster=sec_to_cluster,
            cluster_to_sec=cluster_to_sec,
        )

        return scaled_indicator

class WrappedPercentagePriceOscillator:
    def __init__(self,
                 close_atom_name,
                 clustering_atom_name,
                 window_slow,
                 window_fast,
                 window_sign,
                 rets_form_list,
                 reference,
                 reference_window,
                 windows_buffer_mult):
        self.close_atom_name = close_atom_name
        self.clustering_atom_name = clustering_atom_name
        self.window_slow = window_slow
        self.window_fast = window_fast
        self.window_sign = window_sign
        self.rets_form_list = rets_form_list
        self.reference_window = reference_window
        self.reference = reference
        self.standard_scaler = StandardScaler()
        self.windows_buffer_mult = windows_buffer_mult

    def transform(self,
                  atoms,
                  start_feature_data_date,
                  prediction_date,
                  realization_date,
                  reference_id,
                  active_cross_section_series,
                  universe_name):

        params_list = [prediction_date, 
                       start_feature_data_date,
                       self.close_atom_name,
                       self.window_slow,
                       self.window_fast,
                       self.window_sign,
                       self.rets_form_list,
                       self.reference_window]

        path_to_file = get_path_to_file(
            params_list=params_list, 
            ABS_PATH_TO_CACHE=ABS_PATH_TO_CACHE+universe_name+'/', 
            unique_name='WrappedPercentagePriceOscillator')
            
        indicator = get_pickled_indicator(path_to_file)
        if indicator.empty:
            atom_name_list = [self.close_atom_name,]
            windows_list = [self.window_slow,
                            self.window_fast,
                            self.window_sign,
                            self.reference_window]
                            
            (close_atom,) = get_processed_atoms(
                atoms=atoms, 
                atom_name_list=atom_name_list, 
                start_feature_data_date=start_feature_data_date, 
                prediction_date=prediction_date, 
                windows_list=windows_list, 
                rets_form_list=self.rets_form_list,
                windows_buffer_mult=self.windows_buffer_mult,
                active_cross_section=active_cross_section_series[
                    prediction_date
                ]
            )
            indicator = close_atom.apply(
                lambda x: PercentagePriceOscillator(
                    close=x,
                    window_slow=self.window_slow,
                    window_fast=self.window_fast,
                    window_sign=self.window_sign,
                ).ppo()
            ).iloc[-self.reference_window:].replace({np.inf: np.nan, -np.inf: np.nan}).dropna(how='all', axis=1)
            pickle.dump(indicator, open(path_to_file, 'wb'))

        if self.reference == 'cross':
            (sec_to_cluster, 
            cluster_to_sec) = get_clusters(
                prediction_date=prediction_date, 
                clustering_atom=atoms[self.clustering_atom_name],
                active_cross_section=active_cross_section_series[
                    prediction_date
                ],
                universe_name=universe_name,
            )
        else:
            sec_to_cluster = None
            cluster_to_sec = None

        scaled_indicator = get_scaled_indicator(
            indicator=indicator, 
            scaler=self.standard_scaler, 
            reference=self.reference, 
            reference_window=self.reference_window, 
            reference_id=reference_id,
            sec_to_cluster=sec_to_cluster,
            cluster_to_sec=cluster_to_sec,
        )

        return scaled_indicator

class WrappedPercentageVolumeOscillator:
    def __init__(self,
                 volume_atom_name,
                 clustering_atom_name,
                 window_slow,
                 window_fast,
                 window_sign,
                 rets_form_list,
                 reference,
                 reference_window,
                 windows_buffer_mult):
        self.volume_atom_name = volume_atom_name
        self.clustering_atom_name = clustering_atom_name
        self.window_slow = window_slow
        self.window_fast = window_fast
        self.window_sign = window_sign
        self.rets_form_list = rets_form_list
        self.reference_window = reference_window
        self.reference = reference
        self.standard_scaler = StandardScaler()
        self.windows_buffer_mult = windows_buffer_mult

    def transform(self,
                  atoms,
                  start_feature_data_date,
                  prediction_date,
                  realization_date,
                  reference_id,
                  active_cross_section_series,
                  universe_name):

        params_list = [prediction_date, 
                       start_feature_data_date,
                       self.volume_atom_name,
                       self.window_slow,
                       self.window_fast,
                       self.window_sign,
                       self.rets_form_list,
                       self.reference_window]

        path_to_file = get_path_to_file(
            params_list=params_list, 
            ABS_PATH_TO_CACHE=ABS_PATH_TO_CACHE+universe_name+'/', 
            unique_name='WrappedPercentageVolumeOscillator')
            
        indicator = get_pickled_indicator(path_to_file)
        if indicator.empty:
            atom_name_list = [self.volume_atom_name,]
            windows_list = [self.window_slow,
                            self.window_fast,
                            self.window_sign,
                            self.reference_window]
                            
            (volume_atom,) = get_processed_atoms(
                atoms=atoms, 
                atom_name_list=atom_name_list, 
                start_feature_data_date=start_feature_data_date, 
                prediction_date=prediction_date, 
                windows_list=windows_list, 
                rets_form_list=self.rets_form_list,
                windows_buffer_mult=self.windows_buffer_mult,
                active_cross_section=active_cross_section_series[
                    prediction_date
                ]
            )
            indicator = volume_atom.apply(
                lambda x: PercentageVolumeOscillator(
                    volume=x,
                    window_slow=self.window_slow,
                    window_fast=self.window_fast,
                    window_sign=self.window_sign,
                ).pvo()
            ).iloc[-self.reference_window:].replace({np.inf: np.nan, -np.inf: np.nan}).dropna(how='all', axis=1)
            pickle.dump(indicator, open(path_to_file, 'wb'))

        if self.reference == 'cross':
            (sec_to_cluster, 
            cluster_to_sec) = get_clusters(
                prediction_date=prediction_date, 
                clustering_atom=atoms[self.clustering_atom_name],
                active_cross_section=active_cross_section_series[
                    prediction_date
                ],
                universe_name=universe_name,
            )
        else:
            sec_to_cluster = None
            cluster_to_sec = None

        scaled_indicator = get_scaled_indicator(
            indicator=indicator, 
            scaler=self.standard_scaler, 
            reference=self.reference, 
            reference_window=self.reference_window, 
            reference_id=reference_id,
            sec_to_cluster=sec_to_cluster,
            cluster_to_sec=cluster_to_sec,
        )

        return scaled_indicator

class WrappedROCIndicator:
    def __init__(self,
                 close_atom_name,
                 clustering_atom_name,
                 window,
                 rets_form_list,
                 reference,
                 reference_window,
                 windows_buffer_mult):
        self.close_atom_name = close_atom_name
        self.clustering_atom_name = clustering_atom_name
        self.window = window
        self.rets_form_list = rets_form_list
        self.reference_window = reference_window
        self.reference = reference
        self.standard_scaler = StandardScaler()
        self.windows_buffer_mult = windows_buffer_mult

    def transform(self,
                  atoms,
                  start_feature_data_date,
                  prediction_date,
                  realization_date,
                  reference_id,
                  active_cross_section_series,
                  universe_name):

        params_list = [prediction_date, 
                       start_feature_data_date,
                       self.close_atom_name,
                       self.window,
                       self.rets_form_list,
                       self.reference_window]

        path_to_file = get_path_to_file(
            params_list=params_list, 
            ABS_PATH_TO_CACHE=ABS_PATH_TO_CACHE+universe_name+'/', 
            unique_name='WrappedROCIndicator')
            
        indicator = get_pickled_indicator(path_to_file)
        if indicator.empty:
            atom_name_list = [self.close_atom_name,]
            windows_list = [self.window,
                            self.reference_window]
                            
            (close_atom,) = get_processed_atoms(
                atoms=atoms, 
                atom_name_list=atom_name_list, 
                start_feature_data_date=start_feature_data_date, 
                prediction_date=prediction_date, 
                windows_list=windows_list, 
                rets_form_list=self.rets_form_list,
                windows_buffer_mult=self.windows_buffer_mult,
                active_cross_section=active_cross_section_series[
                    prediction_date
                ]
            )
            indicator = close_atom.apply(
                lambda x: ROCIndicator(
                    close=x,
                    window=self.window,
                ).roc()
            ).iloc[-self.reference_window:].replace({np.inf: np.nan, -np.inf: np.nan}).dropna(how='all', axis=1)
            pickle.dump(indicator, open(path_to_file, 'wb'))

        if self.reference == 'cross':
            (sec_to_cluster, 
            cluster_to_sec) = get_clusters(
                prediction_date=prediction_date, 
                clustering_atom=atoms[self.clustering_atom_name],
                active_cross_section=active_cross_section_series[
                    prediction_date
                ],
                universe_name=universe_name,
            )
        else:
            sec_to_cluster = None
            cluster_to_sec = None

        scaled_indicator = get_scaled_indicator(
            indicator=indicator, 
            scaler=self.standard_scaler, 
            reference=self.reference, 
            reference_window=self.reference_window, 
            reference_id=reference_id,
            sec_to_cluster=sec_to_cluster,
            cluster_to_sec=cluster_to_sec,
        )

        return scaled_indicator

class WrappedRSIIndicator:
    def __init__(self,
                 close_atom_name,
                 clustering_atom_name,
                 window,
                 rets_form_list,
                 reference,
                 reference_window,
                 windows_buffer_mult):
        self.close_atom_name = close_atom_name
        self.clustering_atom_name = clustering_atom_name
        self.window = window
        self.rets_form_list = rets_form_list
        self.reference_window = reference_window
        self.reference = reference
        self.standard_scaler = StandardScaler()
        self.windows_buffer_mult = windows_buffer_mult

    def transform(self,
                  atoms,
                  start_feature_data_date,
                  prediction_date,
                  realization_date,
                  reference_id,
                  active_cross_section_series,
                  universe_name):

        params_list = [prediction_date, 
                       start_feature_data_date,
                       self.close_atom_name,
                       self.window,
                       self.rets_form_list,
                       self.reference_window]

        path_to_file = get_path_to_file(
            params_list=params_list, 
            ABS_PATH_TO_CACHE=ABS_PATH_TO_CACHE+universe_name+'/', 
            unique_name='WrappedRSIIndicator')
            
        indicator = get_pickled_indicator(path_to_file)
        if indicator.empty:
            atom_name_list = [self.close_atom_name,]
            windows_list = [self.window,
                            self.reference_window]
                            
            (close_atom,) = get_processed_atoms(
                atoms=atoms, 
                atom_name_list=atom_name_list, 
                start_feature_data_date=start_feature_data_date, 
                prediction_date=prediction_date, 
                windows_list=windows_list, 
                rets_form_list=self.rets_form_list,
                windows_buffer_mult=self.windows_buffer_mult,
                active_cross_section=active_cross_section_series[
                    prediction_date
                ]
            )
            indicator = close_atom.apply(
                lambda x: RSIIndicator(
                    close=x,
                    window=self.window,
                ).rsi()
            ).iloc[-self.reference_window:].replace({np.inf: np.nan, -np.inf: np.nan}).dropna(how='all', axis=1)
            pickle.dump(indicator, open(path_to_file, 'wb'))

        if self.reference == 'cross':
            (sec_to_cluster, 
            cluster_to_sec) = get_clusters(
                prediction_date=prediction_date, 
                clustering_atom=atoms[self.clustering_atom_name],
                active_cross_section=active_cross_section_series[
                    prediction_date
                ],
                universe_name=universe_name,
            )
        else:
            sec_to_cluster = None
            cluster_to_sec = None

        scaled_indicator = get_scaled_indicator(
            indicator=indicator, 
            scaler=self.standard_scaler, 
            reference=self.reference, 
            reference_window=self.reference_window, 
            reference_id=reference_id,
            sec_to_cluster=sec_to_cluster,
            cluster_to_sec=cluster_to_sec,
        )

        return scaled_indicator

class WrappedStochRSIIndicator:
    def __init__(self,
                 close_atom_name,
                 clustering_atom_name,
                 window,
                 smooth1,
                 smooth2,
                 rets_form_list,
                 reference,
                 reference_window,
                 windows_buffer_mult):
        self.close_atom_name = close_atom_name
        self.clustering_atom_name = clustering_atom_name
        self.window = window
        self.smooth1 = smooth1
        self.smooth2 = smooth2
        self.rets_form_list = rets_form_list
        self.reference_window = reference_window
        self.reference = reference
        self.standard_scaler = StandardScaler()
        self.windows_buffer_mult = windows_buffer_mult

    def transform(self,
                  atoms,
                  start_feature_data_date,
                  prediction_date,
                  realization_date,
                  reference_id,
                  active_cross_section_series,
                  universe_name):

        params_list = [prediction_date, 
                       start_feature_data_date,
                       self.close_atom_name,
                       self.window,
                       self.smooth1,
                       self.smooth2,
                       self.rets_form_list,
                       self.reference_window]

        path_to_file = get_path_to_file(
            params_list=params_list, 
            ABS_PATH_TO_CACHE=ABS_PATH_TO_CACHE+universe_name+'/', 
            unique_name='WrappedStochRSIIndicator')
            
        indicator = get_pickled_indicator(path_to_file)
        if indicator.empty:
            atom_name_list = [self.close_atom_name,]
            windows_list = [self.window,
                            self.smooth1,
                            self.smooth2,
                            self.reference_window]
                            
            (close_atom,) = get_processed_atoms(
                atoms=atoms, 
                atom_name_list=atom_name_list, 
                start_feature_data_date=start_feature_data_date, 
                prediction_date=prediction_date, 
                windows_list=windows_list, 
                rets_form_list=self.rets_form_list,
                windows_buffer_mult=self.windows_buffer_mult,
                active_cross_section=active_cross_section_series[
                    prediction_date
                ]
            )
            indicator = close_atom.apply(
                lambda x: StochRSIIndicator(
                    close=x,
                    window=self.window,
                    smooth1=self.smooth1,
                    smooth2=self.smooth2,
                ).stochrsi()
            ).iloc[-self.reference_window:].replace({np.inf: np.nan, -np.inf: np.nan}).dropna(how='all', axis=1)
            pickle.dump(indicator, open(path_to_file, 'wb'))

        if self.reference == 'cross':
            (sec_to_cluster, 
            cluster_to_sec) = get_clusters(
                prediction_date=prediction_date, 
                clustering_atom=atoms[self.clustering_atom_name],
                active_cross_section=active_cross_section_series[
                    prediction_date
                ],
                universe_name=universe_name,
            )
        else:
            sec_to_cluster = None
            cluster_to_sec = None

        scaled_indicator = get_scaled_indicator(
            indicator=indicator, 
            scaler=self.standard_scaler, 
            reference=self.reference, 
            reference_window=self.reference_window, 
            reference_id=reference_id,
            sec_to_cluster=sec_to_cluster,
            cluster_to_sec=cluster_to_sec,
        )

        return scaled_indicator

class WrappedStochasticOscillator:
    def __init__(self,
                 high_atom_name,
                 low_atom_name,
                 close_atom_name,
                 clustering_atom_name,
                 window,
                 smooth_window,
                 rets_form_list,
                 reference,
                 reference_window,
                 windows_buffer_mult):
        self.high_atom_name = high_atom_name
        self.low_atom_name = low_atom_name
        self.close_atom_name = close_atom_name
        self.clustering_atom_name = clustering_atom_name
        self.window = window
        self.smooth_window = smooth_window
        self.rets_form_list = rets_form_list
        self.reference_window = reference_window
        self.reference = reference
        self.standard_scaler = StandardScaler()
        self.windows_buffer_mult = windows_buffer_mult

    def transform(self,
                  atoms,
                  start_feature_data_date,
                  prediction_date,
                  realization_date,
                  reference_id,
                  active_cross_section_series,
                  universe_name):

        params_list = [prediction_date, 
                       start_feature_data_date,
                       self.high_atom_name,
                       self.low_atom_name,
                       self.close_atom_name,
                       self.window,
                       self.smooth_window,
                       self.rets_form_list,
                       self.reference_window]

        path_to_file = get_path_to_file(
            params_list=params_list, 
            ABS_PATH_TO_CACHE=ABS_PATH_TO_CACHE+universe_name+'/', 
            unique_name='WrappedStochasticOscillator')
            
        indicator = get_pickled_indicator(path_to_file)
        if indicator.empty:
            atom_name_list = [self.high_atom_name, 
                              self.low_atom_name, 
                              self.close_atom_name,]
            windows_list = [self.window,
                            self.smooth_window,
                            self.reference_window]
                            
            (high_atom,
            low_atom,
            close_atom,) = get_processed_atoms(
                atoms=atoms, 
                atom_name_list=atom_name_list, 
                start_feature_data_date=start_feature_data_date, 
                prediction_date=prediction_date, 
                windows_list=windows_list, 
                rets_form_list=self.rets_form_list,
                windows_buffer_mult=self.windows_buffer_mult,
                active_cross_section=active_cross_section_series[
                    prediction_date
                ]
            )
            indicator = close_atom.apply(
                lambda x: StochasticOscillator(
                    high=high_atom[x.name],
                    low=low_atom[x.name],
                    close=x,
                    window=self.window,
                    smooth_window=self.smooth_window,
                ).stoch()
            ).iloc[-self.reference_window:].replace({np.inf: np.nan, -np.inf: np.nan}).dropna(how='all', axis=1)
            pickle.dump(indicator, open(path_to_file, 'wb'))

        if self.reference == 'cross':
            (sec_to_cluster, 
            cluster_to_sec) = get_clusters(
                prediction_date=prediction_date, 
                clustering_atom=atoms[self.clustering_atom_name],
                active_cross_section=active_cross_section_series[
                    prediction_date
                ],
                universe_name=universe_name,
            )
        else:
            sec_to_cluster = None
            cluster_to_sec = None

        scaled_indicator = get_scaled_indicator(
            indicator=indicator, 
            scaler=self.standard_scaler, 
            reference=self.reference, 
            reference_window=self.reference_window, 
            reference_id=reference_id,
            sec_to_cluster=sec_to_cluster,
            cluster_to_sec=cluster_to_sec,
        )

        return scaled_indicator

class WrappedTSIIndicator:
    def __init__(self,
                 close_atom_name,
                 clustering_atom_name,
                 window_slow,
                 window_fast,
                 rets_form_list,
                 reference,
                 reference_window,
                 windows_buffer_mult):
        self.close_atom_name = close_atom_name
        self.clustering_atom_name = clustering_atom_name
        self.window_slow = window_slow
        self.window_fast = window_fast
        self.rets_form_list = rets_form_list
        self.reference_window = reference_window
        self.reference = reference
        self.standard_scaler = StandardScaler()
        self.windows_buffer_mult = windows_buffer_mult

    def transform(self,
                  atoms,
                  start_feature_data_date,
                  prediction_date,
                  realization_date,
                  reference_id,
                  active_cross_section_series,
                  universe_name):

        params_list = [prediction_date, 
                       start_feature_data_date,
                       self.close_atom_name,
                       self.window_slow,
                       self.window_fast,
                       self.rets_form_list,
                       self.reference_window]

        path_to_file = get_path_to_file(
            params_list=params_list, 
            ABS_PATH_TO_CACHE=ABS_PATH_TO_CACHE+universe_name+'/', 
            unique_name='WrappedTSIIndicator')
            
        indicator = get_pickled_indicator(path_to_file)
        if indicator.empty:
            atom_name_list = [self.close_atom_name,]
            windows_list = [self.window_slow,
                            self.window_fast,
                            self.reference_window]
                            
            (close_atom,) = get_processed_atoms(
                atoms=atoms, 
                atom_name_list=atom_name_list, 
                start_feature_data_date=start_feature_data_date, 
                prediction_date=prediction_date, 
                windows_list=windows_list, 
                rets_form_list=self.rets_form_list,
                windows_buffer_mult=self.windows_buffer_mult,
                active_cross_section=active_cross_section_series[
                    prediction_date
                ]
            )
            indicator = close_atom.apply(
                lambda x: TSIIndicator(
                    close=x,
                    window_slow=self.window_slow,
                    window_fast=self.window_fast,
                ).tsi()
            ).iloc[-self.reference_window:].replace({np.inf: np.nan, -np.inf: np.nan}).dropna(how='all', axis=1)
            pickle.dump(indicator, open(path_to_file, 'wb'))

        if self.reference == 'cross':
            (sec_to_cluster, 
            cluster_to_sec) = get_clusters(
                prediction_date=prediction_date, 
                clustering_atom=atoms[self.clustering_atom_name],
                active_cross_section=active_cross_section_series[
                    prediction_date
                ],
                universe_name=universe_name,
            )
        else:
            sec_to_cluster = None
            cluster_to_sec = None

        scaled_indicator = get_scaled_indicator(
            indicator=indicator, 
            scaler=self.standard_scaler, 
            reference=self.reference, 
            reference_window=self.reference_window, 
            reference_id=reference_id,
            sec_to_cluster=sec_to_cluster,
            cluster_to_sec=cluster_to_sec,
        )

        return scaled_indicator

class WrappedUltimateOscillator:
    def __init__(self,
                 high_atom_name,
                 low_atom_name,
                 close_atom_name,
                 clustering_atom_name,
                 window1,
                 window2,
                 window3,
                 weight1,
                 weight2,
                 weight3,
                 rets_form_list,
                 reference,
                 reference_window,
                 windows_buffer_mult):
        self.high_atom_name = high_atom_name
        self.low_atom_name = low_atom_name
        self.close_atom_name = close_atom_name
        self.clustering_atom_name = clustering_atom_name
        self.window1 = window1
        self.window2 = window2
        self.window3 = window3
        self.weight1 = weight1
        self.weight2 = weight2
        self.weight3 = weight3
        self.rets_form_list = rets_form_list
        self.reference_window = reference_window
        self.reference = reference
        self.standard_scaler = StandardScaler()
        self.windows_buffer_mult = windows_buffer_mult

    def transform(self,
                  atoms,
                  start_feature_data_date,
                  prediction_date,
                  realization_date,
                  reference_id,
                  active_cross_section_series,
                  universe_name):

        params_list = [prediction_date, 
                       start_feature_data_date,
                       self.high_atom_name,
                       self.low_atom_name,
                       self.close_atom_name,
                       self.window1,
                       self.window2,
                       self.window3,
                       self.weight1,
                       self.weight2,
                       self.weight3,
                       self.rets_form_list,
                       self.reference_window]

        path_to_file = get_path_to_file(
            params_list=params_list, 
            ABS_PATH_TO_CACHE=ABS_PATH_TO_CACHE+universe_name+'/', 
            unique_name='WrappedUltimateOscillator')
            
        indicator = get_pickled_indicator(path_to_file)
        if indicator.empty:
            atom_name_list = [self.high_atom_name, 
                              self.low_atom_name, 
                              self.close_atom_name,]
            windows_list = [self.window1,
                            self.window2,
                            self.window3,
                            self.reference_window]
                            
            (high_atom,
            low_atom,
            close_atom,) = get_processed_atoms(
                atoms=atoms, 
                atom_name_list=atom_name_list, 
                start_feature_data_date=start_feature_data_date, 
                prediction_date=prediction_date, 
                windows_list=windows_list, 
                rets_form_list=self.rets_form_list,
                windows_buffer_mult=self.windows_buffer_mult,
                active_cross_section=active_cross_section_series[
                    prediction_date
                ]
            )
            indicator = close_atom.apply(
                lambda x: UltimateOscillator(
                    high=high_atom[x.name],
                    low=low_atom[x.name],
                    close=x,
                    window1=self.window1,
                    window2=self.window2,
                    window3=self.window3,
                    weight1=self.weight1,
                    weight2=self.weight2,
                    weight3=self.weight3,
                ).ultimate_oscillator()
            ).iloc[-self.reference_window:].replace({np.inf: np.nan, -np.inf: np.nan}).dropna(how='all', axis=1)
            pickle.dump(indicator, open(path_to_file, 'wb'))

        if self.reference == 'cross':
            (sec_to_cluster, 
            cluster_to_sec) = get_clusters(
                prediction_date=prediction_date, 
                clustering_atom=atoms[self.clustering_atom_name],
                active_cross_section=active_cross_section_series[
                    prediction_date
                ],
                universe_name=universe_name,
            )
        else:
            sec_to_cluster = None
            cluster_to_sec = None

        scaled_indicator = get_scaled_indicator(
            indicator=indicator, 
            scaler=self.standard_scaler, 
            reference=self.reference, 
            reference_window=self.reference_window, 
            reference_id=reference_id,
            sec_to_cluster=sec_to_cluster,
            cluster_to_sec=cluster_to_sec,
        )

        return scaled_indicator

class WrappedWilliamsRIndicator:
    def __init__(self,
                 high_atom_name,
                 low_atom_name,
                 close_atom_name,
                 clustering_atom_name,
                 lbp,
                 rets_form_list,
                 reference,
                 reference_window,
                 windows_buffer_mult):
        self.high_atom_name = high_atom_name
        self.low_atom_name = low_atom_name
        self.close_atom_name = close_atom_name
        self.clustering_atom_name = clustering_atom_name
        self.lbp = lbp
        self.rets_form_list = rets_form_list
        self.reference_window = reference_window
        self.reference = reference
        self.standard_scaler = StandardScaler()
        self.windows_buffer_mult = windows_buffer_mult

    def transform(self,
                  atoms,
                  start_feature_data_date,
                  prediction_date,
                  realization_date,
                  reference_id,
                  active_cross_section_series,
                  universe_name):

        params_list = [prediction_date, 
                       start_feature_data_date,
                       self.high_atom_name,
                       self.low_atom_name,
                       self.close_atom_name,
                       self.lbp,
                       self.rets_form_list,
                       self.reference_window]

        path_to_file = get_path_to_file(
            params_list=params_list, 
            ABS_PATH_TO_CACHE=ABS_PATH_TO_CACHE+universe_name+'/', 
            unique_name='WrappedWilliamsRIndicator')
            
        indicator = get_pickled_indicator(path_to_file)
        if indicator.empty:
            atom_name_list = [self.high_atom_name, 
                              self.low_atom_name, 
                              self.close_atom_name,]
            windows_list = [self.lbp,
                            self.reference_window]
                            
            (high_atom,
            low_atom,
            close_atom,) = get_processed_atoms(
                atoms=atoms, 
                atom_name_list=atom_name_list, 
                start_feature_data_date=start_feature_data_date, 
                prediction_date=prediction_date, 
                windows_list=windows_list, 
                rets_form_list=self.rets_form_list,
                windows_buffer_mult=self.windows_buffer_mult,
                active_cross_section=active_cross_section_series[
                    prediction_date
                ]
            )
            indicator = close_atom.apply(
                lambda x: WilliamsRIndicator(
                    high=high_atom[x.name],
                    low=low_atom[x.name],
                    close=x,
                    lbp=self.lbp,
                ).williams_r()
            ).iloc[-self.reference_window:].replace({np.inf: np.nan, -np.inf: np.nan}).dropna(how='all', axis=1)
            pickle.dump(indicator, open(path_to_file, 'wb'))

        if self.reference == 'cross':
            (sec_to_cluster, 
            cluster_to_sec) = get_clusters(
                prediction_date=prediction_date, 
                clustering_atom=atoms[self.clustering_atom_name],
                active_cross_section=active_cross_section_series[
                    prediction_date
                ],
                universe_name=universe_name,
            )
        else:
            sec_to_cluster = None
            cluster_to_sec = None

        scaled_indicator = get_scaled_indicator(
            indicator=indicator, 
            scaler=self.standard_scaler, 
            reference=self.reference, 
            reference_window=self.reference_window, 
            reference_id=reference_id,
            sec_to_cluster=sec_to_cluster,
            cluster_to_sec=cluster_to_sec,
        )

        return scaled_indicator

