import pickle
import numpy as np
import pandas as pd
from ta.volume import *
from sklearn.preprocessing import StandardScaler

from ..utils import *

class WrappedAccDistIndexIndicator:
    def __init__(self,
                 high_atom_name,
                 low_atom_name,
                 close_atom_name,
                 volume_atom_name,
                 clustering_atom_name,
                 rets_form_list,
                 reference,
                 reference_window,
                 windows_buffer_mult):
        self.high_atom_name = high_atom_name
        self.low_atom_name = low_atom_name
        self.close_atom_name = close_atom_name
        self.volume_atom_name = volume_atom_name
        self.clustering_atom_name = clustering_atom_name
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
                       self.volume_atom_name,
                       self.rets_form_list,
                       self.reference_window]

        path_to_file = get_path_to_file(
            params_list=params_list, 
            ABS_PATH_TO_CACHE=ABS_PATH_TO_CACHE+universe_name+'/', 
            unique_name='WrappedAccDistIndexIndicator')

        indicator = get_pickled_indicator(path_to_file)
        if indicator.empty:
            atom_name_list = [self.high_atom_name,
                              self.low_atom_name,
                              self.close_atom_name,
                              self.volume_atom_name,]
            windows_list = [self.reference_window]
                            
            (high_atom, 
            low_atom,
            close_atom,
            volume_atom) = get_processed_atoms(
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
                lambda x: AccDistIndexIndicator(
                    high=x,
                    low=low_atom[x.name],
                    close=close_atom[x.name],
                    volume=volume_atom[x.name],
                ).acc_dist_index()
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

class WrappedChaikinMoneyFlowIndicator:
    def __init__(self,
                 high_atom_name,
                 low_atom_name,
                 close_atom_name,
                 volume_atom_name,
                 clustering_atom_name,
                 window,
                 rets_form_list,
                 reference,
                 reference_window,
                 windows_buffer_mult):
        self.high_atom_name = high_atom_name
        self.low_atom_name = low_atom_name
        self.close_atom_name = close_atom_name
        self.volume_atom_name = volume_atom_name
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
                       self.volume_atom_name,
                       self.window,
                       self.rets_form_list,
                       self.reference_window]

        path_to_file = get_path_to_file(
            params_list=params_list, 
            ABS_PATH_TO_CACHE=ABS_PATH_TO_CACHE+universe_name+'/', 
            unique_name='WrappedChaikinMoneyFlowIndicator')

        indicator = get_pickled_indicator(path_to_file)
        if indicator.empty:
            atom_name_list = [self.high_atom_name,
                              self.low_atom_name,
                              self.close_atom_name,
                              self.volume_atom_name,]
            windows_list = [self.window,
                            self.reference_window]
                            
            (high_atom, 
            low_atom,
            close_atom,
            volume_atom) = get_processed_atoms(
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
                lambda x: ChaikinMoneyFlowIndicator(
                    high=x,
                    low=low_atom[x.name],
                    close=close_atom[x.name],
                    volume=volume_atom[x.name],
                    window=self.window,
                ).chaikin_money_flow()
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

class WrappedEaseOfMovementIndicator:
    def __init__(self,
                 high_atom_name,
                 low_atom_name,
                 volume_atom_name,
                 clustering_atom_name,
                 window,
                 rets_form_list,
                 reference,
                 reference_window,
                 windows_buffer_mult):
        self.high_atom_name = high_atom_name
        self.low_atom_name = low_atom_name
        self.volume_atom_name = volume_atom_name
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
                       self.volume_atom_name,
                       self.window,
                       self.rets_form_list,
                       self.reference_window]

        path_to_file = get_path_to_file(
            params_list=params_list, 
            ABS_PATH_TO_CACHE=ABS_PATH_TO_CACHE+universe_name+'/', 
            unique_name='WrappedEaseOfMovementIndicator')

        indicator = get_pickled_indicator(path_to_file)
        if indicator.empty:
            atom_name_list = [self.high_atom_name,
                              self.low_atom_name,
                              self.volume_atom_name,]
            windows_list = [self.window,
                            self.reference_window]
                            
            (high_atom, 
            low_atom,
            volume_atom) = get_processed_atoms(
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
                lambda x: EaseOfMovementIndicator(
                    high=x,
                    low=low_atom[x.name],
                    volume=volume_atom[x.name],
                    window=self.window,
                ).ease_of_movement()
            ).iloc[-self.reference_window:].replace({np.inf: np.nan, -np.inf: np.nan}).dropna(how='all', axis=1).replace(
                {np.inf: np.nan, -np.inf: np.nan}
            ).ffill()
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

class WrappedForceIndexIndicator:
    def __init__(self,
                 close_atom_name,
                 volume_atom_name,
                 clustering_atom_name,
                 window,
                 rets_form_list,
                 reference,
                 reference_window,
                 windows_buffer_mult):
        self.close_atom_name = close_atom_name
        self.volume_atom_name = volume_atom_name
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
                       self.volume_atom_name,
                       self.window,
                       self.rets_form_list,
                       self.reference_window]

        path_to_file = get_path_to_file(
            params_list=params_list, 
            ABS_PATH_TO_CACHE=ABS_PATH_TO_CACHE+universe_name+'/', 
            unique_name='WrappedForceIndexIndicator')

        indicator = get_pickled_indicator(path_to_file)
        if indicator.empty:
            atom_name_list = [self.close_atom_name,
                              self.volume_atom_name,]
            windows_list = [self.window,
                            self.reference_window]
                            
            (close_atom,
            volume_atom) = get_processed_atoms(
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
                lambda x: ForceIndexIndicator(
                    close=x,
                    volume=volume_atom[x.name],
                    window=self.window,
                ).force_index()
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

class WrappedMFIIndicator:
    def __init__(self,
                 high_atom_name,
                 low_atom_name,
                 close_atom_name,
                 volume_atom_name,
                 clustering_atom_name,
                 window,
                 rets_form_list,
                 reference,
                 reference_window,
                 windows_buffer_mult):
        self.high_atom_name = high_atom_name
        self.low_atom_name = low_atom_name
        self.close_atom_name = close_atom_name
        self.volume_atom_name = volume_atom_name
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
                       self.volume_atom_name,
                       self.window,
                       self.rets_form_list,
                       self.reference_window]

        path_to_file = get_path_to_file(
            params_list=params_list, 
            ABS_PATH_TO_CACHE=ABS_PATH_TO_CACHE+universe_name+'/', 
            unique_name='WrappedMFIIndicator')

        indicator = get_pickled_indicator(path_to_file)
        if indicator.empty:
            atom_name_list = [self.high_atom_name,
                              self.low_atom_name,
                              self.close_atom_name,
                              self.volume_atom_name,]
            windows_list = [self.window,
                            self.reference_window]
                            
            (high_atom, 
            low_atom,
            close_atom,
            volume_atom) = get_processed_atoms(
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
                lambda x: MFIIndicator(
                    high=x,
                    low=low_atom[x.name],
                    close=close_atom[x.name],
                    volume=volume_atom[x.name],
                    window=self.window,
                ).money_flow_index()
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

class WrappedNegativeVolumeIndexIndicator:
    def __init__(self,
                 close_atom_name,
                 volume_atom_name,
                 clustering_atom_name,
                 rets_form_list,
                 reference,
                 reference_window,
                 windows_buffer_mult):
        self.close_atom_name = close_atom_name
        self.volume_atom_name = volume_atom_name
        self.clustering_atom_name = clustering_atom_name
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
                       self.volume_atom_name,
                       self.rets_form_list,
                       self.reference_window]

        path_to_file = get_path_to_file(
            params_list=params_list, 
            ABS_PATH_TO_CACHE=ABS_PATH_TO_CACHE+universe_name+'/', 
            unique_name='WrappedNegativeVolumeIndexIndicator')

        indicator = get_pickled_indicator(path_to_file)
        if indicator.empty:
            atom_name_list = [self.close_atom_name,
                              self.volume_atom_name,]
            windows_list = [self.reference_window]
                            
            (close_atom,
            volume_atom) = get_processed_atoms(
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
                lambda x: NegativeVolumeIndexIndicator(
                    close=x,
                    volume=volume_atom[x.name],
                ).negative_volume_index()
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

class WrappedOnBalanceVolumeIndicator:
    def __init__(self,
                 close_atom_name,
                 volume_atom_name,
                 clustering_atom_name,
                 rets_form_list,
                 reference,
                 reference_window,
                 windows_buffer_mult):
        self.close_atom_name = close_atom_name
        self.volume_atom_name = volume_atom_name
        self.clustering_atom_name = clustering_atom_name
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
                       self.volume_atom_name,
                       self.rets_form_list,
                       self.reference_window]

        path_to_file = get_path_to_file(
            params_list=params_list, 
            ABS_PATH_TO_CACHE=ABS_PATH_TO_CACHE+universe_name+'/', 
            unique_name='WrappedOnBalanceVolumeIndicator')

        indicator = get_pickled_indicator(path_to_file)
        if indicator.empty:
            atom_name_list = [self.close_atom_name,
                              self.volume_atom_name,]
            windows_list = [self.reference_window]
                            
            (close_atom,
            volume_atom) = get_processed_atoms(
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
                lambda x: OnBalanceVolumeIndicator(
                    close=x,
                    volume=volume_atom[x.name],
                ).on_balance_volume()
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

class WrappedVolumePriceTrendIndicator:
    def __init__(self,
                 close_atom_name,
                 volume_atom_name,
                 clustering_atom_name,
                 rets_form_list,
                 reference,
                 reference_window,
                 windows_buffer_mult):
        self.close_atom_name = close_atom_name
        self.volume_atom_name = volume_atom_name
        self.clustering_atom_name = clustering_atom_name
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
                       self.volume_atom_name,
                       self.rets_form_list,
                       self.reference_window]

        path_to_file = get_path_to_file(
            params_list=params_list, 
            ABS_PATH_TO_CACHE=ABS_PATH_TO_CACHE+universe_name+'/', 
            unique_name='WrappedVolumePriceTrendIndicator')

        indicator = get_pickled_indicator(path_to_file)
        if indicator.empty:
            atom_name_list = [self.close_atom_name,
                              self.volume_atom_name,]
            windows_list = [self.reference_window]
                            
            (close_atom,
            volume_atom) = get_processed_atoms(
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
                lambda x: VolumePriceTrendIndicator(
                    close=x,
                    volume=volume_atom[x.name],
                ).volume_price_trend()
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
