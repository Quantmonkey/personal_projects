import os.path
ABS_PATH_TO_CACHE='C:/Users/Oscar/git/transparent_offchain/main/models/features/cache/'

import pickle
import numpy as np
import pandas as pd
from ta.trend import *
from sklearn.preprocessing import StandardScaler

from ..utils import *

class WrappedADXIndicator:
    def __init__(self,
                 high_atom_name,
                 low_atom_name,
                 close_atom_name,
                 clustering_atom_name,
                 window,
                 rets_form_list,
                 reference,
                 reference_window,
                 windows_buffer_mult):
        self.high_atom_name = high_atom_name
        self.low_atom_name = low_atom_name
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
                       self.high_atom_name,
                       self.low_atom_name,
                       self.close_atom_name,
                       self.window,
                       self.rets_form_list,
                       self.reference_window]

        path_to_file = get_path_to_file(
            params_list=params_list, 
            ABS_PATH_TO_CACHE=ABS_PATH_TO_CACHE+universe_name+'/', 
            unique_name='WrappedADXIndicator')

        indicator = get_pickled_indicator(path_to_file)
        if indicator.empty:
            atom_name_list = [self.high_atom_name,
                              self.low_atom_name,
                              self.close_atom_name,]
            windows_list = [self.window,
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
                lambda x: ADXIndicator(
                    high=high_atom[x.name],
                    low=low_atom[x.name],
                    close=x,
                    window=self.window
                ).adx()
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

class WrappedAroonIndicator:
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
            unique_name='WrappedAroonIndicator')

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
                lambda x: AroonIndicator(
                    close=x,
                    window=self.window
                ).aroon_indicator()
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

class WrappedCCIIndicator:
    def __init__(self,
                 high_atom_name,
                 low_atom_name,
                 close_atom_name,
                 clustering_atom_name,
                 window,
                 constant,
                 rets_form_list,
                 reference,
                 reference_window,
                 windows_buffer_mult):
        self.high_atom_name = high_atom_name
        self.low_atom_name = low_atom_name
        self.close_atom_name = close_atom_name
        self.clustering_atom_name = clustering_atom_name
        self.window = window
        self.constant = constant
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
                       self.constant,
                       self.rets_form_list,
                       self.reference_window]

        path_to_file = get_path_to_file(
            params_list=params_list, 
            ABS_PATH_TO_CACHE=ABS_PATH_TO_CACHE+universe_name+'/', 
            unique_name='WrappedCCIIndicator')

        indicator = get_pickled_indicator(path_to_file)
        if indicator.empty:
            atom_name_list = [
                self.high_atom_name,
                self.low_atom_name,
                self.close_atom_name,
            ]
            windows_list = [self.window,
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
                lambda x: CCIIndicator(
                    high=high_atom[x.name],
                    low=low_atom[x.name],
                    close=x,
                    window=self.window,
                    constant=self.constant,
                ).cci()
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

class WrappedDPOIndicator:
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
            unique_name='WrappedDPOIndicator')

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
                lambda x: DPOIndicator(
                    close=x,
                    window=self.window
                ).dpo()
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

class WrappedKSTIndicator:
    def __init__(self,
                 close_atom_name,
                 clustering_atom_name,
                 roc1,
                 roc2,
                 roc3,
                 roc4,
                 window1,
                 window2,
                 window3,
                 window4,
                 nsig,
                 rets_form_list,
                 reference,
                 reference_window,
                 windows_buffer_mult):
        self.close_atom_name = close_atom_name
        self.clustering_atom_name = clustering_atom_name
        self.roc1 = roc1
        self.roc2 = roc2
        self.roc3 = roc3
        self.roc4 = roc4
        self.window1 = window1
        self.window2 = window2
        self.window3 = window3
        self.window4 = window4
        self.nsig = nsig
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
                       self.roc1,
                       self.roc2,
                       self.roc3,
                       self.roc4,
                       self.window1,
                       self.window2,
                       self.window3,
                       self.window4,
                       self.nsig,
                       self.rets_form_list,
                       self.reference_window]

        path_to_file = get_path_to_file(
            params_list=params_list, 
            ABS_PATH_TO_CACHE=ABS_PATH_TO_CACHE+universe_name+'/', 
            unique_name='WrappedKSTIndicator')

        indicator = get_pickled_indicator(path_to_file)
        if indicator.empty:
            atom_name_list = [self.close_atom_name,]
            windows_list = [self.roc1,
                            self.roc2,
                            self.roc3,
                            self.roc4,
                            self.window1,
                            self.window2,
                            self.window3,
                            self.window4,
                            self.nsig,
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
                lambda x: KSTIndicator(
                    close=x,
                    roc1=self.roc1,
                    roc2=self.roc2,
                    roc3=self.roc3,
                    roc4=self.roc4,
                    window1=self.window1,
                    window2=self.window2,
                    window3=self.window3,
                    window4=self.window4,
                    nsig=self.nsig,
                ).kst()
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

class WrappedMACD:
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
            unique_name='WrappedMACD')

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
                lambda x: MACD(
                    close=x,
                    window_slow=self.window_slow,
                    window_fast=self.window_fast,
                    window_sign=self.window_sign,
                ).macd()
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

class WrappedMassIndex:
    def __init__(self,
                 high_atom_name,
                 low_atom_name,
                 clustering_atom_name,
                 window_slow,
                 window_fast,
                 rets_form_list,
                 reference,
                 reference_window,
                 windows_buffer_mult):
        self.high_atom_name = high_atom_name
        self.low_atom_name = low_atom_name
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
                       self.high_atom_name,
                       self.low_atom_name,
                       self.window_slow,
                       self.window_fast,
                       self.rets_form_list,
                       self.reference_window]

        path_to_file = get_path_to_file(
            params_list=params_list, 
            ABS_PATH_TO_CACHE=ABS_PATH_TO_CACHE+universe_name+'/', 
            unique_name='WrappedMACD')

        indicator = get_pickled_indicator(path_to_file)
        if indicator.empty:
            atom_name_list = [
                self.high_atom_name,
                self.low_atom_name,
            ]
            windows_list = [self.window_slow,
                            self.window_fast,
                            self.reference_window]
                            
            (high_atom,
            low_atom) = get_processed_atoms(
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
                lambda x: MassIndex(
                    high=x,
                    low=low_atom[x.name],
                    window_slow=self.window_slow,
                    window_fast=self.window_fast,
                ).mass_index()
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

class WrappedPSARIndicator:
    def __init__(self,
                 high_atom_name,
                 low_atom_name,
                 close_atom_name,
                 clustering_atom_name,
                 step,
                 max_step,
                 rets_form_list,
                 reference,
                 reference_window,
                 windows_buffer_mult):
        self.high_atom_name = high_atom_name
        self.low_atom_name = low_atom_name
        self.close_atom_name = close_atom_name
        self.clustering_atom_name = clustering_atom_name
        self.step = step
        self.max_step = max_step
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
                       self.step,
                       self.max_step,
                       self.rets_form_list,
                       self.reference_window]

        path_to_file = get_path_to_file(
            params_list=params_list, 
            ABS_PATH_TO_CACHE=ABS_PATH_TO_CACHE+universe_name+'/', 
            unique_name='WrappedPSARIndicator')

        indicator = get_pickled_indicator(path_to_file)
        if indicator.empty:
            atom_name_list = [
                self.high_atom_name,
                self.low_atom_name,
                self.close_atom_name,
            ]
            windows_list = [self.reference_window]
                            
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
                lambda x: PSARIndicator(
                    high=high_atom[x.name],
                    low=low_atom[x.name],
                    close=x,
                    step=self.step,
                    max_step=self.max_step,
                ).psar()
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

class WrappedSTCIndicator:
    def __init__(self,
                 close_atom_name,
                 clustering_atom_name,
                 window_slow,
                 window_fast,
                 cycle,
                 smooth1,
                 smooth2,
                 rets_form_list,
                 reference,
                 reference_window,
                 windows_buffer_mult):
        self.close_atom_name = close_atom_name
        self.clustering_atom_name = clustering_atom_name
        self.window_slow = window_slow
        self.window_fast = window_fast
        self.cycle = cycle
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
                       self.window_slow,
                       self.window_fast,
                       self.cycle,
                       self.smooth1,
                       self.smooth2,
                       self.rets_form_list,
                       self.reference_window]

        path_to_file = get_path_to_file(
            params_list=params_list, 
            ABS_PATH_TO_CACHE=ABS_PATH_TO_CACHE+universe_name+'/', 
            unique_name='WrappedSTCIndicator')

        indicator = get_pickled_indicator(path_to_file)
        if indicator.empty:
            atom_name_list = [self.close_atom_name,]
            windows_list = [self.window_slow,
                            self.window_fast,
                            self.cycle,
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
                lambda x: STCIndicator(
                    close=x,
                    window_slow=self.window_slow,
                    window_fast=self.window_fast,
                    cycle=self.cycle,
                    smooth1=self.smooth1,
                    smooth2=self.smooth2,
                ).stc()
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

class WrappedTRIXIndicator:
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
            unique_name='WrappedTRIXIndicator')

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
                lambda x: TRIXIndicator(
                    close=x,
                    window=self.window
                ).trix()
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

class WrappedVortexIndicator:
    def __init__(self,
                 high_atom_name,
                 low_atom_name,
                 close_atom_name,
                 clustering_atom_name,
                 window,
                 rets_form_list,
                 reference,
                 reference_window,
                 windows_buffer_mult):
        self.high_atom_name = high_atom_name
        self.low_atom_name = low_atom_name
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
                       self.high_atom_name,
                       self.low_atom_name,
                       self.close_atom_name,
                       self.window,
                       self.rets_form_list,
                       self.reference_window]

        path_to_file = get_path_to_file(
            params_list=params_list, 
            ABS_PATH_TO_CACHE=ABS_PATH_TO_CACHE+universe_name+'/', 
            unique_name='WrappedVortexIndicator')

        indicator = get_pickled_indicator(path_to_file)
        if indicator.empty:
            atom_name_list = [
                self.high_atom_name,
                self.low_atom_name,
                self.close_atom_name,
            ]
            windows_list = [self.window,
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
                lambda x: VortexIndicator(
                    high=high_atom[x.name],
                    low=low_atom[x.name],
                    close=x,
                    window=self.window,
                ).vortex_indicator_diff()
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