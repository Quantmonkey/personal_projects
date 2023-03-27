import os.path
ABS_PATH_TO_CACHE='C:/Users/Oscar/git/transparent_offchain/main/models/features/cache/'

import pickle
import numpy as np
import pandas as pd
from ta.trend import *
from sklearn.preprocessing import StandardScaler

from numba import jit, njit, prange

from ..utils import *

class RollingMoment:
    def __init__(self,
                 close_rets_atom_name,
                 clustering_atom_name,
                 moment,
                 side,
                 window,
                 rets_form_list,
                 reference,
                 reference_window,
                 windows_buffer_mult):
        self.close_rets_atom_name = close_rets_atom_name
        self.clustering_atom_name = clustering_atom_name
        self.moment = moment
        self.side = side
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
                       self.close_rets_atom_name,
                       self.moment,
                       self.side,
                       self.window,
                       self.rets_form_list,
                       self.reference_window]

        path_to_file = get_path_to_file(
            params_list=params_list, 
            ABS_PATH_TO_CACHE=ABS_PATH_TO_CACHE+universe_name+'/', 
            unique_name='RollingMoment')

        indicator = get_pickled_indicator(path_to_file)
        if indicator.empty:
            atom_name_list = [self.close_rets_atom_name,]
            windows_list = [self.reference_window]
                            
            (close_rets_atom,) = get_processed_atoms(
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

            indicator = close_rets_atom.apply(
                lambda x: self.get_moment_series(
                    close_rets=x,
                    moment=self.moment,
                    side=self.side,
                    window=self.window,
                )
            ).iloc[-self.reference_window:].replace({np.inf: np.nan, -np.inf: np.nan}).dropna(how='all', axis=1).fillna(0)
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
    
    def get_moment_series(self,
                          close_rets,
                          moment,
                          side,
                          window):
        
        if side == 'downside':
            close_rets = close_rets[close_rets < 0]
        elif side == 'upside':
            close_rets = close_rets[close_rets > 0]

        if moment == 'mean':
            moment_series = close_rets.rolling(window).mean()
        elif moment == 'std':
            moment_series = close_rets.rolling(window).std()
        elif moment == 'skew':
            moment_series = close_rets.rolling(window).skew()
        elif moment == 'kurt':
            moment_series = close_rets.rolling(window).apply(
                lambda x: pd.Series.kurtosis(x)
            )
        elif moment == 'autocorr':
            moment_series = close_rets.rolling(window).apply(
                lambda x: pd.Series.autocorr(x)
            )

        return moment_series